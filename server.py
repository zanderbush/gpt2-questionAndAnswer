from transformers import AutoModelWithLMHead, AutoTokenizer, top_k_top_p_filtering
import torch
from flask import Flask, request, Response, jsonify
from flask import Flask, render_template, request, Response, send_file, jsonify

from torch.nn import functional as F
from queue import Queue, Empty
import time
import threading

# Server & Handling Setting
app = Flask(__name__)

requests_queue = Queue()
BATCH_SIZE = 1
CHECK_INTERVAL = 0.1

tokenizer = AutoTokenizer.from_pretrained("zanderbush/Intellectual")
model = AutoModelWithLMHead.from_pretrained("zanderbush/Intellectual", return_dict=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Queue 핸들링
def handle_requests_by_batch():
    while True:
        requests_batch = []
        while not (len(requests_batch) >= BATCH_SIZE):
            try:
                requests_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))
            except Empty:
                continue

            for requests in requests_batch:
                requests['output'] = run_model(requests['input'][0])


# 쓰레드
threading.Thread(target=handle_requests_by_batch).start()

def run_model(prompt, num=1, length=30):
    try:
        prompt = prompt.strip()
        text = tokenizer.encode(prompt)
        myinput, past = torch.tensor([text]), None
        logits, past = model(myinput, past = past)
        logits = logits[0,-1]
        probabilities = torch.nn.functional.softmax(logits)
        best_logits, best_indices = logits.topk(50)
        best_words = [tokenizer.decode([idx.item()]) for idx in best_indices]
        text.append(best_indices[0].item())
        best_probabilities = probabilities[best_indices].tolist()
        generated_texts = []
        for i in range(50):
            f = ('{}'.format(best_words[i]))
            generated_texts.append(f)
        print(generated_texts)
        return generated_texts

    except Exception as e:
        print(e)
        return 500



@app.route("/api/", methods=['GET'])
def generate():

    if requests_queue.qsize() > BATCH_SIZE:
        return jsonify({'error': 'Too Many Requests'}), 429

    try:
        args = []

        keyword=request.args.get('keyword')

        args.append(keyword)

    except Exception:
        print("Empty Text")
        return Response("fail", status=400)

    req = {
        'input': args
    }
    requests_queue.put(req)

    while 'output' not in req:
        time.sleep(CHECK_INTERVAL)

    return req['output']


# Health Check

@app.route('/healthz')
def health():
    return "ok", 200

@app.route('/')
def main():
    return render_template('index.html')

if __name__ == "__main__":
    from waitress import serve
    serve(app, host='0.0.0.0', port=80)
