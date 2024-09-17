from flaskr import app
from flask import render_template, request, redirect, url_for

from transformers import pipeline
import torch
import whisperx
import gc
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import os

transcriptString = ''

### Transcription section
def performASR(filename, numSpeakers):
    # set up
    device = "cpu"
    batch_size = 4
    compute_type = "int8"
    if torch.cuda.is_available():
        device = "cuda"
        batch_size = 32
        compute_type = "float16"
    audio_file = filename # name of the audio file

    # create model
    audio = whisperx.load_audio(audio_file)
    model = whisperx.load_model("large-v3", device, compute_type=compute_type)

    # transcribe
    result = model.transcribe(audio, batch_size=batch_size)

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model
    
    # 2. Align whisper output
    language = result["language"]
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    # audio diarization
    # provide a user authorization token to "auth_token"
    diarize_model = whisperx.DiarizationPipeline(use_auth_token="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", device=device)
    diarize_segments = diarize_model(audio, num_speakers=numSpeakers)
    # print(diarize_segments.speaker.unique())

    result = whisperx.assign_word_speakers(diarize_segments, result)
    # print(diarize_segments)
    # print(result["segments"]) # segments are now assigned speaker IDs
    return result, language

def convertIntoStringEnglish(result):
    s = ''
    transcript = []
    currSpeaker = ''
    flag = True
    for x in result['word_segments']:
        if flag:
            currSpeaker = x['speaker']
            flag = False
        if len(list(x)) == 5 and currSpeaker != x['speaker']:
            transcript.append(currSpeaker + ': ' + s)
            s = ''
            currSpeaker = x['speaker']
        s += x['word'] + ' '
    transcript.append(currSpeaker + ': ' + s)
    return transcript

def convertIntoStringJapanese(result):
    s = ''
    transcript = []
    currSpeaker = ''
    flag = True
    for x in result['word_segments']:
        if flag:
            currSpeaker = x['speaker']
            flag = False
        if len(list(x)) == 5 and currSpeaker != x['speaker']:
            transcript.append(currSpeaker + ': ' + s)
            s = ''
            currSpeaker = x['speaker']
        s += x['word']
    transcript.append(currSpeaker + ': ' + s)
    return transcript

def convertIntoStringEnglishString(result):
    s = ''
    transcript = ''
    currSpeaker = ''
    flag = True
    for x in result['word_segments']:
        if flag:
            currSpeaker = x['speaker']
            flag = False
        if len(list(x)) == 5 and currSpeaker != x['speaker']:
            transcript += currSpeaker + ': ' + s + '\n'
            s = ''
            currSpeaker = x['speaker']
        s += x['word'] + ' '
    transcript += currSpeaker + ': ' + s
    return transcript

def convertIntoStringJapaneseString(result):
    s = ''
    transcript = ''
    currSpeaker = ''
    flag = True
    for x in result['word_segments']:
        if flag:
            currSpeaker = x['speaker']
            flag = False
        if len(list(x)) == 5 and currSpeaker != x['speaker']:
            transcript += currSpeaker + ': ' + s + '\n'
            s = ''
            currSpeaker = x['speaker']
        s += x['word']
    transcript += currSpeaker + ': ' + s + '\n'
    return transcript

def summarizeEnglish(transcript, maxLength):
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    ARTICLE_TO_SUMMARIZE = (transcript)
    inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="pt")
    summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=maxLength)
    return tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

### Web App section
@app.route('/')
def index():
    return render_template(
        'index.html'
    )

@app.route('/transcribe', methods=['POST'])
def transcribe():
    global transcriptString
    audio = request.files['audio']
    if audio.filename == '':
        return render_template(
        'index.html'
        )
    audio.save(f'{audio.filename}')
    num_speaker = int(request.form['num_speaker'])
    result, language = performASR(audio.filename, num_speaker)
    transcript = ''
    if language == 'ja':
        transcript = convertIntoStringJapanese(result)
        transcriptString = convertIntoStringJapaneseString(result)
    else:
        transcript = convertIntoStringEnglish(result)
        transcriptString = convertIntoStringEnglishString(result)
    os.remove(audio.filename)
    return render_template(
        'transcribe.html',
        transcript = transcript
    )

@app.route('/summarization')
def summarization():
    summary = summarizeEnglish(transcriptString, 100)
    return render_template(
        'summary.html',
        summary = summary
    )
