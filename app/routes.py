from flask import Flask, render_template, request, jsonify
from app import app
from app.lang_modules import langfunction

# 간단한 홈페이지 렌더링
@app.route('/')
def index():
    return {"message": "In Progress"}

# 채팅 API 엔드포인트
@app.route('/api/chat', methods=['POST'])
def chat():
    user_input = request.json['user_input']
    try: 
        bot_output = langfunction(user_input)
    except: 
        bot_output = 'error : false input'

    return jsonify({'bot_output': bot_output})