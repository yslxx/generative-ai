# sentiment analysis demo
# ! pip install transformers 
# ! pip install gradio 
# https://huggingface.co/matthewburke/korean_sentiment 모델 사용 
from transformers import pipeline 
import gradio as gr 

classifier = pipeline('text-classification', model = 'matthewburke/korean_sentiment')

def pred_sentiment(text):
    preds = classifier(text, return_all_scores = True)
    if preds[0][1]['score'] > 0.5:
        return '긍정'
    else:
        return '부정'
    

iface = gr.Interface(fn = pred_sentiment,
                     inputs = gr.inputs.Textbox(lines = 5, placeholder = '감성 분석할 텍스트를 입력해 주세요.'),
                     outputs = 'text',
                     title = '한글 감성 분석',
                     description = '분석하여 긍정(positive)인지 부정(negative)인지 알려줄게요.')

iface.launch(share = True)
    



