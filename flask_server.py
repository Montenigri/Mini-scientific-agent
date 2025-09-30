import flask
from mini_scientific_agent import msa

app = flask.Flask(__name__)
agent = msa()

@app.route('/ask', methods=['POST'])
def ask():
    
    data = flask.request.json
    print("Received request")
    print(data)
    messages = data.get('messages', [])
    query = {"messages": messages}
    thread_id = data.get('thread_id', 'default')
    config = data.get('config', {"configurable": {"thread_id": thread_id}})

    if not query:
        return flask.jsonify({"error": "No query provided"}), 400

    def generate():
        try:
            for chunk in agent.get_streaming_answer(query, config):
                yield chunk
        except Exception as e:
            yield f"[ERROR] {str(e)}"

    return flask.Response(
        flask.stream_with_context(generate()),
        mimetype="text/plain"
    )

@app.route('/')
def index():
    return(flask.render_template('index.html'))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False) # Debug falso per evitare problemi con i thread appesi
