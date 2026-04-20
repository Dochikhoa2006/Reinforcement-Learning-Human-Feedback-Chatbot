from transformers import RobertaTokenizer, RobertaModel
from pyspark.sql import SparkSession
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch


spark = SparkSession.builder.appName ('ReadParquet').getOrCreate ()
dataset = spark.read.parquet ('/Users/chikhoado/Desktop/PROJECTS/RL Human Feedback/ultra_feedback.parquet')


class Reward_Modeling_using_RoBBERTa (nn.Module):

    def __init__ (self):
        super ().__init__ ()

        if torch.backends.mps.is_available ():
            self.hardware_place = torch.device ('mps')
        elif torch.cuda.is_available ():
            self.hardware_place = torch.device ('cuda')
        else:
            self.hardware_place = torch.device ('cpu')

        self.backbone = RobertaModel.from_pretrained ('roberta-base') 
        self.pair_wise_ranking_head = nn.Linear (self.backbone.config.hidden_size, 1)
        self.tokenizer = RobertaTokenizer.from_pretrained ('roberta-base')

        self.to (self.hardware_place)

    def train_model (self, file, optimizer, loss_func):

        batch_size = 4

        file_modified = []
        for row in file.select ('prompt', 'chosen', 'rejected').toLocalIterator ():
            file_modified.append (list (row))
        
        file_prepared = DataLoader (dataset = file_modified, batch_size = batch_size, shuffle = True)

        self.train ()
        for index, file_batch in enumerate (file_prepared):
            input_ids_chosen_batch, attention_mask_chosen_batch, input_ids_rejected_batch, attention_mask_rejected_batch = self.prepare_for_input_training (file_batch)

            input_ids_chosen_batch = torch.tensor (input_ids_chosen_batch, dtype = torch.long).to (self.hardware_place)
            attention_mask_chosen_batch = torch.tensor (attention_mask_chosen_batch, dtype = torch.long).to (self.hardware_place)
            input_ids_rejected_batch = torch.tensor (input_ids_rejected_batch, dtype = torch.long).to (self.hardware_place)
            attention_mask_rejected_batch = torch.tensor (attention_mask_rejected_batch, dtype = torch.long).to (self.hardware_place)

            score_chosen = self.forward (input_ids_chosen_batch, attention_mask_chosen_batch)
            score_rejected = self.forward (input_ids_rejected_batch, attention_mask_rejected_batch)

            deterministic_score = torch.ones_like (score_chosen).to (self.hardware_place)
            loss = loss_func (score_chosen, score_rejected, deterministic_score)

            optimizer.zero_grad ()
            loss.backward ()
            optimizer.step ()

            print (f'Batch Completed: {index + 1} -> Loss: {loss.item ():.4f}')

    def forward (self, processed_input, mask_self_attetion):

        encoder_block_process = self.backbone (input_ids = processed_input, 
                                                attention_mask = mask_self_attetion)
        last_encoder_block_output = encoder_block_process.last_hidden_state

        mask_self_attetion = mask_self_attetion[ : , : , None]
        mask_self_attetion = mask_self_attetion.expand (last_encoder_block_output.size ()).float ()
        non_pad_last_encoder_block_output = last_encoder_block_output * mask_self_attetion

        sum_embeddings = torch.sum (non_pad_last_encoder_block_output, dim = 1)
        number_of_non_pad_tokens = torch.clamp (mask_self_attetion.sum (dim = 1), min = 1e-9) 
        mean_pool_output = sum_embeddings / number_of_non_pad_tokens
        
        pair_wise_value = self.pair_wise_ranking_head (mean_pool_output)
        return pair_wise_value

    def tokenize (self, Prompt, Answer):

        prompt_answer_pair = self.tokenizer (Prompt, Answer, max_length = 511, padding = 'max_length', truncation = True)
        input_ids = prompt_answer_pair['input_ids']
        attention_mask = prompt_answer_pair['attention_mask']

        return [input_ids, attention_mask]

    def prepare_for_input_training (self, file_batch):

        input_ids_chosen, attention_mask_chosen = [], []
        input_ids_rejected, attention_mask_rejected = [], []

        prompt_column = file_batch[0]
        chosen_column = file_batch[1]
        rejected_column = file_batch[2]
        
        for index_row in range (len (prompt_column)):

            Prompt = prompt_column[index_row]
            Chosen = chosen_column[1][0][index_row]
            Chosen_Role = chosen_column[1][1][index_row]
            Rejected = rejected_column[1][0][index_row]
            Rejected_Role = rejected_column[1][1][index_row]

            Chosen_structure = f'[{Chosen_Role}: {Chosen}'
            Rejected_structure = f'[{Rejected_Role}: {Rejected}'

            input_ids_and_attention_mask_chosen_row_i = self.tokenize (Prompt, Chosen_structure)
            input_ids_and_attention_mask_rejected_row_i = self.tokenize (Prompt, Rejected_structure)

            input_ids_chosen.append (input_ids_and_attention_mask_chosen_row_i[0])
            attention_mask_chosen.append (input_ids_and_attention_mask_chosen_row_i[1])

            input_ids_rejected.append (input_ids_and_attention_mask_rejected_row_i[0])
            attention_mask_rejected.append (input_ids_and_attention_mask_rejected_row_i[1])
        
        return input_ids_chosen, attention_mask_chosen, input_ids_rejected, attention_mask_rejected

    def prepare_input_for_inference (self, Prompt, Answer):

        tokenized_prompt_answer_pair = self.tokenize (Prompt, Answer)
        return tokenized_prompt_answer_pair[0], tokenized_prompt_answer_pair[1]

    def predict (self, Prompt, Answer):

        input_ids_prompt_answer, attention_mask_prompt_answer = self.prepare_input_for_inference (Prompt, Answer)
        input_ids_prompt_answer = torch.tensor (input_ids_prompt_answer, dtype = torch.long).unsqueeze (0).to (self.hardware_place)
        attention_mask_prompt_answer = torch.tensor (attention_mask_prompt_answer, dtype = torch.long).unsqueeze (0).to (self.hardware_place)

        self.eval ()
        with torch.no_grad ():
            score_prompt_answer = self.forward (input_ids_prompt_answer, attention_mask_prompt_answer)
        
        self.train ()
        return score_prompt_answer.item ()
    
    def save_model (self):

        self.backbone.save_pretrained ('RM')
        self.tokenizer.save_pretrained ('RM')
        torch.save (self.pair_wise_ranking_head.state_dict (), 'RM/pairwise_ranking_head.pt')
    
    def load_model (self):

        self.backbone = RobertaModel.from_pretrained ('RM')
        self.tokenizer = RobertaTokenizer.from_pretrained ('RM')
        
        pair_wise_ranking_head_loaded = torch.load ('RM/pairwise_ranking_head.pt', map_location = self.hardware_place)
        self.pair_wise_ranking_head.load_state_dict (pair_wise_ranking_head_loaded)

        self.to (self.hardware_place)


if __name__ == '__main__':

    model = Reward_Modeling_using_RoBBERTa ()

    optimizer = optim.Adam (model.parameters ())
    loss_func = nn.MarginRankingLoss ()

    model.train_model (dataset, optimizer, loss_func)
    model.save_model ()










# cd '/Users/chikhoado/Desktop/PROJECTS/RL Human Feedback'
# /opt/homebrew/bin/python3.12 -m venv .venv
# source .venv/bin/activate
# pip install pyspark transformers torch
# python '/Users/chikhoado/Desktop/PROJECTS/RL Human Feedback/Reward_Modeling.py'