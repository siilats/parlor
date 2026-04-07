import base64, requests

audio_b64 = base64.b64encode(open('0.wav', 'rb').read()).decode()
ref_text  = open('0.txt').read().strip()

resp = requests.post('http://127.0.0.1:8013/v1/tts', json={
    'text': 'Clone this voice and say something.',
    'references': [{'audio': audio_b64, 'text': ref_text}],
})
open('cloned.wav', 'wb').write(resp.content)
