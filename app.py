import openai
import json

openai.api_key = "your_openai_api_key"

""" Build a function to call GPT-4 API """
def get_response(prompt, model="text-curie-002", max_tokens=150):
    response = openai.Completion.create(
        engine=model, #or "your_chosen_engine",
        prompt=prompt,
        max_tokens=max_tokens,
        n=5,
        stop=None,
        temperature=0.8,
    )

    return response.choices[0].text.strip()


""" Upload the customized dataset on the OpenAI server """
with open("my_dataset.json", "r") as f:
   data = json.load(f)

dataset = openai.Dataset.create(
   data=data,
   name="my_dataset_name",
   description="my_dataset_metadata",
)

""" Fine tune the OpenAI server based on the uploaded dataset """
fine_tuning = openai.FineTune.create(
   model="text-davinci-002",  
   dataset=dataset["id"], 
   n_steps=1000,  
   prompt_tokens=1024,  
)

# You can check the status of the fine-tuning job using the fine-tuning ID
job_id = fine_tuning["id"]