from transformers import GPT2Tokenizer, GPT2Model
from pyspark.sql import SparkSession
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch


spark = SparkSession.builder.appName ('ReadParquet').getOrCreate ()
dataset = spark.read.parquet ('/Users/chikhoado/Desktop/PROJECTS/RL Human Feedback/ultra_feedback.parquet')


class SFT_using_GPT_2 (nn.Module):

    def __init__ (self):
        super ().__init__ ()

        if torch.backends.mps.is_available ():
            self.hardware_place = torch.device ('mps')
        elif torch.cuda.is_available ():
            self.hardware_place = torch.device ('cuda')
        else:
            self.hardware_place = torch.device ('cpu')

        self.backbone = GPT2Model.from_pretrained ('gpt2')

        self.tokenizer = GPT2Tokenizer.from_pretrained ('gpt2')
        self.tokenizer_built_in ()

        self.policy_head = nn.Linear (self.backbone.config.n_embd, len (self.tokenizer))
        self.value_head = nn.Linear (self.backbone.config.n_embd, 1)
        self.initial_weight_bias_value_head ()

        self.to (self.hardware_place)

    def tokenizer_built_in (self):

        self.tokenizer.padding_side = 'left' 
        
        special_tokens = {'pad_token': '[PAD]', 'cls_token': '[CLS]', 'sep_token': '[SEP]', 'bos_token': '[BOS]'}
        nums_token_added = self.tokenizer.add_special_tokens (special_tokens)

        self.backbone.resize_token_embeddings (len (self.tokenizer))
        original_vocab_size = len (self.tokenizer) - nums_token_added

        all_token_embeddings = self.backbone.get_input_embeddings ().weight.data
        mean_old_token_embeddings = all_token_embeddings[ : original_vocab_size].mean (dim = 0)

        for index in range (nums_token_added):
            position = original_vocab_size + index
            all_token_embeddings[position] = mean_old_token_embeddings

    def initial_weight_bias_value_head (self):

        nn.init.normal_ (self.value_head.weight, mean = 0.0, std = 0.001)
        nn.init.constant_ (self.value_head.bias, 0.0)

    def train_model (self, file, optimizer, loss_func):

        batch_size = 4
        file_pandas = file.select ('prompt', 'chosen').toPandas ()
        file_modified = file_pandas.values.tolist ()
        file_prepared = DataLoader (dataset = file_modified, batch_size = batch_size, shuffle = True)

        self.train ()
        for index, file_batch in enumerate (file_prepared):
            input_ids_batch, masked_attention_batch = self.prepare_input_for_training (file_batch)

            hardward_place_input_ids_batch = torch.tensor (input_ids_batch, dtype = torch.long).to (self.hardware_place)
            hardward_place_masked_attention_batch = torch.tensor (masked_attention_batch, dtype = torch.long).to (self.hardware_place)

            predicted_logits, _ = self.forward (hardward_place_input_ids_batch, hardward_place_masked_attention_batch)
            
            flatten_predicted_logits = predicted_logits[ : , : -1, : ].reshape (-1, len (self.tokenizer))
            flatten_target_index = hardward_place_input_ids_batch[ : , 1 : ].reshape (-1)
            
            loss = loss_func (flatten_predicted_logits, flatten_target_index)

            optimizer.zero_grad ()
            loss.backward ()
            optimizer.step ()

            print (f'Batch Completed: {index + 1} -> Loss: {round (loss.item (), 4)}')

    def forward (self, processed_input, mask_self_attention):

        decoder_block_process = self.backbone (input_ids = processed_input, 
                                                attention_mask = mask_self_attention)
        last_decoder_block_output = decoder_block_process.last_hidden_state
        
        logits_policy_head = self.policy_head (last_decoder_block_output)
        full_return_value_head = self.value_head (last_decoder_block_output)

        return logits_policy_head, full_return_value_head
    
    def tokenize (self, text, padding_decision):

        text_into_tokens = self.tokenizer (text, max_length = 511, padding = padding_decision, truncation = True)
        token_ids = text_into_tokens['input_ids']
        token_masked_attention = text_into_tokens['attention_mask']

        return [token_ids, token_masked_attention]

    def prepare_input_for_training (self, file_batch):

        CLS = self.tokenizer.cls_token
        BOS = self.tokenizer.bos_token
        SEP = self.tokenizer.sep_token
        EOS = self.tokenizer.eos_token

        input_ids_collection = []
        masked_attention_collection = []
        
        prompt_column = file_batch[0]
        chosen_column = file_batch[1]

        for index_row in range (len (prompt_column)):

            Prompt = prompt_column[index_row]
            Answer = chosen_column[1][0][index_row]
            Answer_Role = chosen_column[1][1][index_row]

            input_structure = f"{CLS} {BOS} {Prompt} {SEP} {BOS} [{Answer_Role}]: {Answer} {EOS} {SEP}"
            input_tokenized = self.tokenize (input_structure, 'max_length')

            input_ids_collection.append (input_tokenized[0])
            masked_attention_collection.append (input_tokenized[1])

        return input_ids_collection, masked_attention_collection

    def prepare_input_for_inference (self, Prompt):

        CLS = self.tokenizer.cls_token
        BOS = self.tokenizer.bos_token
        SEP = self.tokenizer.sep_token

        input_structure = f"{CLS} {BOS} {Prompt} {SEP} {BOS}"
        input_tokenized = self.tokenize (input_structure, False)

        return input_tokenized[0]

    def predict (self, user_question, boolean):
        
        tokenized_user_question_ids = self.prepare_input_for_inference (user_question)
        tokenized_user_question_ids = torch.tensor (tokenized_user_question_ids, dtype = torch.long).unsqueeze (0)
        tokenized_user_question_ids = tokenized_user_question_ids.to (self.hardware_place)

        self.eval ()
        answer_ids = []
        temperature = 0.8
        top_k = 50

        with torch.no_grad ():
            self.eval ()
            for _ in range (511):
                
                masked_self_attention = torch.ones_like (tokenized_user_question_ids).to (self.hardware_place)
                sequence_output_logits, _ = self.forward (tokenized_user_question_ids, masked_self_attention)
                last_output_logits = sequence_output_logits[ : , -1, : ]

                if boolean == True:
                    last_output_logits = last_output_logits / temperature
                    predicted_token_id = torch.argmax (last_output_logits, dim = 1)

                    answer_ids.append (predicted_token_id.item ())
                    correct_shape_predicted_token_id = predicted_token_id[ : , None]
                    tokenized_user_question_ids = torch.concat ([tokenized_user_question_ids, correct_shape_predicted_token_id], dim = 1)

                else:
                    top_k_output_logits, _ = torch.topk (last_output_logits, top_k)
                    min_top_k_output_logits = top_k_output_logits[ : , -1].unsqueeze (-1)
                    last_output_logits[last_output_logits < min_top_k_output_logits] = -float ('Inf')
                    last_output_logits = last_output_logits / temperature

                    probs = torch.softmax (last_output_logits, dim = 1)
                    predicted_token_id = torch.multinomial (probs, num_samples = 1)

                    answer_ids.append (predicted_token_id.item ())
                    tokenized_user_question_ids = torch.concat ([tokenized_user_question_ids, predicted_token_id], dim = 1)

                if tokenized_user_question_ids.size (1) >= 1022:
                    break
                if predicted_token_id.item () == self.tokenizer.eos_token_id or predicted_token_id.item () == self.tokenizer.sep_token_id:
                    break

            if len (answer_ids) > 0:
                answer = self.tokenizer.decode (answer_ids, skip_special_tokens = True)
            else:
                answer_ids = "There is no answer"

        self.train ()
        return answer

    def save_model (self, file_path):

        GPT2 = file_path
        policy = file_path + '/policy_head.pt'
        value = file_path + '/value_head.pt'

        self.backbone.save_pretrained (GPT2)
        self.tokenizer.save_pretrained (GPT2)
        torch.save (self.policy_head.state_dict (), policy)
        torch.save (self.value_head.state_dict (), value)

    def load_model (self, file_path):

        GPT2 = file_path 
        policy = file_path + '/policy_head.pt'
        value = file_path + '/value_head.pt'

        self.backbone = GPT2Model.from_pretrained (GPT2)
        self.tokenizer = GPT2Tokenizer.from_pretrained (GPT2)

        policy_head_from_pretrained = torch.load (policy, map_location = self.hardware_place)
        self.policy_head.load_state_dict (policy_head_from_pretrained)

        value_head_from_pretrained = torch.load (value, map_location = self.hardware_place)
        self.value_head.load_state_dict (value_head_from_pretrained)
        
        self.to (self.hardware_place)


if __name__ == '__main__':

    model = SFT_using_GPT_2 ()

    parameters_curernt_training = [{'params': model.backbone.parameters ()},
                                    {'params': model.policy_head.parameters ()}]
    optimizer = optim.Adamax (parameters_curernt_training)
    loss_func = nn.CrossEntropyLoss (ignore_index = model.tokenizer.pad_token_id)

    model.train_model (dataset, optimizer, loss_func)
    model.save_model ('SFT')








# cd '/Users/chikhoado/Desktop/PROJECTS/RL Human Feedback'
# /opt/homebrew/bin/python3.12 -m venv .venv
# source .venv/bin/activate
# pip install pyspark transformers torch
# python '/Users/chikhoado/Desktop/PROJECTS/RL Human Feedback/Supervised_Fine_Tuning.py'