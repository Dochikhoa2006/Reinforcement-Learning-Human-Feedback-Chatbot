from Supervised_Fine_Tuning import SFT_using_GPT_2
from Reward_Modeling import Reward_Modeling_using_RoBBERTa
from torch.utils.data import DataLoader
from openai import OpenAI
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import pandas as pd
import os

load_dotenv ()
openai_key = os.getenv ('OPENAI_API_KEY')
client = OpenAI (api_key = openai_key)

dataset = pd.read_excel ('/Users/chikhoado/Desktop/PROJECTS/RL Human Feedback/Prompt_Dataset.xlsx')
prompt_dataset = dataset['Prompts'].values.tolist ()
prompt_loader = DataLoader (dataset = prompt_dataset, batch_size = 1, shuffle = False)


def LLM_judge_PPO (Prompt, Answer_GPT2, Answer_GPT2_and_PPO):

    response = client.chat.completions.create (
            model = "gpt-4o-mini",
            messages = [
                {"role": "system", "content": "You are an unbiased evaluator."},
                {"role": "user", "content": f"""
    Question: {Prompt}
    Answer A: {Answer_GPT2}
    Answer B: {Answer_GPT2_and_PPO}

    Which is better? Reply ONLY with one characte: A or B."""}])
    
    judgement_answer = response.choices[0]
    judgement_answer = judgement_answer.message
    judgement_answer = judgement_answer.content.strip ()

    return judgement_answer

def LLM_judge_RoBERTa (Prompt, Answer):
        
    response = client.chat.completions.create (
            model = "gpt-4o-mini",
            temperature = 0, 
            messages = [
                {"role": "system", "content": "You are a strict evaluator."},
                {"role": "user", "content": f"""
    Question: {Prompt}
    Answer: {Answer}

    Rate the answer from 1 to 5 based on overall quality.
    1 = very poor, 5 = excellent.

    Reply with ONLY a number (1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75 or 5)."""}])
    
    return response.choices[0].message.content.strip ()

def evaluate (GPT2, GPT2_and_PPO, RoBERTa):

    GPT2_record = 0
    GPT2_and_PPO_record = 0
    RoBERTa_score_track = []

    for Prompt in prompt_loader:
        Answer_GPT2 = GPT2.predict (Prompt, True)
        Answer_GPT2_and_PPO = GPT2_and_PPO.predict (Prompt, True)
        Score_RoBERTa = RoBERTa.predict (Prompt, Answer_GPT2_and_PPO)

        model_better = LLM_judge_PPO (Prompt, Answer_GPT2, Answer_GPT2_and_PPO)
        try:
            if model_better == 'A':
                GPT2_record += 1
            else:
                GPT2_and_PPO_record += 1
        except:
            print ('Answer of LLM is produced in wrong format')

        model_score = LLM_judge_RoBERTa (Prompt, Answer_GPT2_and_PPO)
        try:
            model_score = float (model_score)
            absolute_error = abs (model_score - Score_RoBERTa)
            absolute_error = round (absolute_error, 2)
            RoBERTa_score_track.append (absolute_error)
        except:
            print ('Answer of LLM is produced in wrong format')
    
    return GPT2_record, GPT2_and_PPO_record, RoBERTa_score_track

def plotting (GPT2_record, GPT2_and_PPO_record, RoBERTa_score_track):

    fig, (graph1, graph2) = plt.subplots (1, 2, figsize = (15, 8))

    models = ['GPT-2', 'GPT-2 + PPO']
    values = [GPT2_record, GPT2_and_PPO_record]
    colors = ['skyblue', 'deepskyblue']

    graph1.bar (models, values, color = colors)
    graph1.plot (RoBERTa_score_track)
    graph1.set_title ('LLM Judge: Preference Comparison')
    graph1.set_ylabel ('Total Wins')

    graph2.plot (RoBERTa_score_track, color = 'skyblue')
    graph2.set_title ('RoBERTa Evaluated by LLM Judgement')
    graph2.set_ylabel ('Absolute Error')
    graph2.set_xlabel ('Number of Samples')
    graph2.grid (True)

    plt.tight_layout ()
    plt.savefig ('RLHF_Evaluation.png')
    plt.show ()


if __name__ == '__main__':

    GPT2 = SFT_using_GPT_2 ()
    GPT2.load_model ('SFT')

    GPT2_and_PPO = SFT_using_GPT_2 ()
    GPT2_and_PPO.load_model ('RL')

    RoBERTa = Reward_Modeling_using_RoBBERTa ()
    RoBERTa.load_model ()

    GPT2_record, GPT2_and_PPO_record, RoBERTa_score_track = evaluate (GPT2, GPT2_and_PPO, RoBERTa)
    plotting (GPT2_record, GPT2_and_PPO_record, RoBERTa_score_track)