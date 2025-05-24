# agents.py

# 1) Load environment early
from dotenv import load_dotenv
import os

# point at your env file
load_dotenv(dotenv_path='claude.env')

# 2) CrewAI imports
from crewai import Agent, Message

# Agent: Captures video frames & detects emotion
class EmotionAgent(Agent):
    def __init__(self):
        super().__init__(
            name="emotion_agent",
            role="Emotion Sensor",
            goal="Continuously detect the user's facial emotion from webcam frames and share the dominant emotion.",
            backstory="I process camera input to infer emotional state using the DeepFace model.",
            verbose=True,
        )

    def run(self, message: Message) -> Message:
        frame = message.data['frame']
        # … your existing emotion detection logic here …
        data = emotion_detector(frame)       # pseudocode
        transcript = data.get('transcript', '')
        emotion = data.get('emotion', 'neutral')

        # Build the prompt
        prompt = (
            f"The user just said: '{transcript}'.\n"
            f"Detected emotion: {emotion}.\n"
            "Please respond in an empathetic, supportive manner appropriate to the emotion."
        )

        # 3) Call Claude via OpenAI wrapper, using the env-loaded key
        import openai
        openai.api_key = os.getenv("ANTHROPIC_API_KEY")

        response = openai.ChatCompletion.create(
            model='claude-v1',
            messages=[{'role': 'user', 'content': prompt}],
        )
        reply = response.choices[0].message['content']

        return Message(
            sender=self.name,
            receiver=message.sender,
            data={'response': reply},
        )

# … other Agent and Coordinator definitions …

if __name__ == "__main__":
    coord = Coordinator()
    coord.run()
