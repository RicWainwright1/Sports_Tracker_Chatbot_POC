import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Check if ELEVEN_API_KEY environment variable is set
eleven_api_key = os.environ.get("ELEVEN_API_KEY")

print("Environment variable check:")
if eleven_api_key:
    print(f"✅ ELEVEN_API_KEY is set (starts with: {eleven_api_key[:4]}***)")
    
    try:
        # Try to import ElevenLabs
        from elevenlabs.client import ElevenLabs
        print("✅ ElevenLabs package is installed")
        
        # Try to create client
        client = ElevenLabs(api_key=eleven_api_key)
        print("✅ ElevenLabs client created")
        
        # Try to get voices
        voices = client.voices.get_all()
        print(f"✅ Successfully connected to ElevenLabs API. Found {len(voices.voices)} voices.")
        
    except ImportError as e:
        print(f"❌ Error importing ElevenLabs package: {e}")
    except Exception as e:
        print(f"❌ Error connecting to ElevenLabs API: {e}")
else:
    print("❌ ELEVEN_API_KEY environment variable is not set")
    print("Please set the ELEVEN_API_KEY environment variable with your ElevenLabs API key")
    print("You can do this by:")
    print("1. Creating a .env file in the project root with: ELEVEN_API_KEY=your_api_key_here")
    print("2. Or setting the environment variable directly in your terminal:")
    print("   - On macOS/Linux: export ELEVEN_API_KEY=your_api_key_here")
    print("   - On Windows: set ELEVEN_API_KEY=your_api_key_here") 