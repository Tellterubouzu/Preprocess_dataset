from openai import OpenAI
import os

MODEL_PRICING = {
    "gpt-4.1": {"prompt": 2.00, "cached": 0.50, "completion": 8.00},
    "gpt-4.1-mini": {"prompt": 0.40, "cached": 0.10, "completion": 1.60},
    "gpt-4.1-nano": {"prompt": 0.10, "cached": 0.025, "completion": 0.40},
    "gpt-4.5-preview": {"prompt": 75.00, "cached": 37.50, "completion": 150.00},
    "gpt-4o": {"prompt": 2.50, "cached": 1.25, "completion": 10.00},
    "gpt-4o-2024-08-06": {"prompt": 2.50, "cached": 1.25, "completion": 10.00},
    "gpt-4o-mini": {"prompt": 0.15, "cached": 0.075, "completion": 0.60},
    "o1": {"prompt": 15.00, "cached": 7.50, "completion": 60.00},
    "o1-pro": {"prompt": 150.00, "cached": None, "completion": 600.00},
    "o3": {"prompt": 10.00, "cached": 2.50, "completion": 40.00},
    "o4-mini": {"prompt": 1.10, "cached": 0.275, "completion": 4.40},
    "o3-mini": {"prompt": 1.10, "cached": 0.55, "completion": 4.40},
    "o1-mini": {"prompt": 1.10, "cached": 0.55, "completion": 4.40},
}

class OpenAI_API:
    def __init__(self, model="gpt-4o-2024-08-06", api_key=os.getenv("OPENAI_API_KEY"),system_prompt_file=None,max_tokens=1000,temperature=0.5):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.messages = []
        self.system_prompt_file = system_prompt_file
        if system_prompt_file:
            self.system_prompt = self.load_system_prompt(system_prompt_file)
        else:
            self.system_prompt = None
        self.max_tokens = max_tokens
        self.temperature = temperature

    def load_system_prompt(self, prompt_file):
        with open(prompt_file, "r") as f:
            sys_prompt = f.read()
        self.messages.append({"role": "system", "content": sys_prompt})
        
    def generate_text(self, prompt):
        self.messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=self.model, 
            messages=self.messages,
            max_tokens=self.max_tokens,
            temperature = self.temperature
            )
        content = response.choices[0].message.content
        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        price_info = MODEL_PRICING.get(self.model, {"prompt": 0.0, "completion": 0.0})
        cost_usd = (prompt_tokens / 1000000) * price_info["prompt"] + (completion_tokens / 1000000) * price_info["completion"]
        self.messages.append({"role": "assistant", "content": content})
        return content, cost_usd
    
    def reset_conversation(self):
        self.messages = []
        self.load_system_prompt(self.system_prompt_file)

    def add_to_conversation(self, role, content):
        self.conversation.append({"role": role, "content": content})

    def get_conversation(self):
        return self.conversation
    
    def cancell_conversation(self,num_messages=1):
        self.messages = self.messages[:-num_messages]
    