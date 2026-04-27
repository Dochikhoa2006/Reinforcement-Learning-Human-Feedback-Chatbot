from Supervised_Fine_Tuning import SFT_using_GPT_2
from Reward_Modeling import Reward_Modeling_using_RoBBERTa
from pyspark.sql import SparkSession
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch


spark = SparkSession.builder.appName ('ReadParquet').getOrCreate ()
dataset = spark.read.parquet ('/Users/chikhoado/Desktop/PROJECTS/RL Human Feedback/ultra_feedback.parquet')


class Proximal_Policy_Optimization :

    def __init__(self):
        
        self.SFT = SFT_using_GPT_2 ()
        self.RM = Reward_Modeling_using_RoBBERTa ()

        self.SFT.load_model ('SFT')
        self.RM.load_model ()

    def prepare_for_evaluate (self, Prompt, Answer):

        CLS = self.SFT.tokenizer.cls_token
        BOS = self.SFT.tokenizer.bos_token
        SEP = self.SFT.tokenizer.sep_token
        EOS = self.SFT.tokenizer.eos_token

        input_structure = f"{CLS} {BOS} {Prompt} {SEP} {BOS} {Answer} {EOS}"
        token_ids, token_masked_attention = self.SFT.tokenize (input_structure, False)

        token_ids = torch.tensor (token_ids).to (self.SFT.hardware_place)
        token_masked_attention = torch.tensor (token_masked_attention).to (self.SFT.hardware_place)
        
        token_ids_generation = token_ids[1 : , None]
        token_ids = token_ids.unsqueeze (0)
        token_masked_attention = token_masked_attention.unsqueeze (0)
        
        return token_ids_generation, token_ids, token_masked_attention
    
    def evaluate (self, token_ids_generation, token_ids, token_masked_attention):

        logits_policy_head, full_return_value_head = self.SFT.forward (token_ids, token_masked_attention)

        logits_policy_head = logits_policy_head[0, : -1, : ]
        full_return_value_head = full_return_value_head[0, : -1, 0]

        log_probs = nn.functional.log_softmax (logits_policy_head, dim = 1)
        log_probs = torch.gather (log_probs, index = token_ids_generation, dim = 1)
        log_probs = log_probs[ : , 0]

        return log_probs, full_return_value_head

    def train_model (self, file, optimizer, loss_func_value_head):
        
        batch_size = 1
        mini_iteration = 4
        epsilon = 0.2
        beta = 5e-3

        file_pandas = file.select ('prompt').toPandas ()
        file_modified = file_pandas.values.tolist ()
        file_prepared = DataLoader (dataset = file_modified, batch_size = batch_size, shuffle = True)

        for tracking, file_batch in enumerate (file_prepared):
            Prompt = file_batch[0][0]

            with torch.no_grad ():
                Answer = self.SFT.predict (Prompt, False)
                terminal_reward_frozen = self.RM.predict (Prompt, Answer)
                token_ids_generation, token_ids, token_masked_attention = self.prepare_for_evaluate (Prompt, Answer)
                log_probs_frozen, _ = self.evaluate (token_ids_generation, token_ids, token_masked_attention)
            
            for _ in range (mini_iteration):
                log_probs_now, V_each_token = self.evaluate (token_ids_generation, token_ids, token_masked_attention)

                log_probs_ratio = torch.exp (log_probs_now - log_probs_frozen)
                log_probs_ratio_clipped = torch.clamp (log_probs_ratio, 1 - epsilon, 1 + epsilon)
                
                G_each_token = - beta * (log_probs_now - log_probs_frozen)
                G_each_token[-1] += terminal_reward_frozen                
                advantage = G_each_token - V_each_token
                advantage = advantage.detach ()

                unclipped_path = log_probs_ratio * advantage
                clipped_path = log_probs_ratio_clipped * advantage

                policy_loss = - torch.min (unclipped_path, clipped_path).mean ()
                value_loss = loss_func_value_head (G_each_token, V_each_token)
                total_loss = policy_loss + 0.05 * value_loss

                optimizer.zero_grad ()
                total_loss.backward ()

                torch.nn.utils.clip_grad_norm_ (self.SFT.parameters(), max_norm = 5.0)
                optimizer.step ()

            print (f'Batch Completed: {tracking + 1} -> Loss: {total_loss.item ():.8f}', flush = True)
    

if __name__ == '__main__':

    model = Proximal_Policy_Optimization ()

    optimizer = optim.Adam (model.SFT.parameters (), lr = 5e-6)
    loss_func_value_head = nn.MSELoss ()

    model.train_model (dataset, optimizer, loss_func_value_head)
    model.SFT.save_model ('RL')