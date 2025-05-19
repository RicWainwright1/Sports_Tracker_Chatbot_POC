import os
from dotenv import load_dotenv
from elevenlabs import play
from elevenlabs.client import ElevenLabs

load_dotenv()
elevenlabs_api_key = os.getenv("ELEVENLABS_KEY")

from elevenlabs.client import ElevenLabs

client = ElevenLabs(
  api_key=elevenlabs_api_key,
)


#response = client.voices.get_all()
#print(response.voices)

audio = client.text_to_speech.convert(
    text="Hey champ! I'm CoachIQ — your personal sports expert powered by The Inisghts Family. Whether you're breaking down stats, comparing trends, or prepping for your next big presentation, I'm here to give you smart, fast, and accurate insights. Ask me anything — let's get you game-ready!",
    voice_id="JBFqnCBsd6RMkjVDRZzb",
    model_id="eleven_multilingual_v2",
    output_format="mp3_44100_128",
)

play(audio)