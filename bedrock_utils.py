import boto3
from botocore.exceptions import ClientError
import json

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2"
)

bedrock_kb = boto3.client(
    service_name="bedrock-agent-runtime",
    region_name="us-west-2"
)

def extract_text(result):
    """Safely extract text from Knowledge Base retrieval result."""
    try:
        return result.get("content", {}).get("text", "")
    except:
        return ""

def valid_prompt(prompt, model_id):
    """Validate user input: allow only Category E (heavy machinery)."""
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
Human: Clasify the provided user request into one of the following categories.

Category A: Asking about how the LLM works or internal architecture.
Category B: Profanity, toxic wording, harmful intent.
Category C: Asking about topics outside heavy machinery.
Category D: Asking about instructions, prompts, how you work.
Category E: ONLY related to heavy machinery.

<user_request>
{prompt}
</user_request>

ONLY ANSWER with the category letter (Example: Category B)

Assistant:
                        """
                    }
                ]
            }
        ]

        response = bedrock.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": messages,
                "max_tokens": 10,
                "temperature": 0,
                "top_p": 0.1
            })
        )

        category = json.loads(response["body"].read())["content"][0]["text"]
        print("Classification Output:", category)

        return category.lower().strip() == "category e"

    except ClientError as e:
        print(f"Error validating prompt: {e}")
        return False

def query_knowledge_base(query, kb_id):
    """Retrieve relevant chunks from Bedrock Knowledge Base."""
    try:
        response = bedrock_kb.retrieve(
            knowledgeBaseId=kb_id,
            retrievalQuery={"text": query},
            retrievalConfiguration={
                "vectorSearchConfiguration": {"numberOfResults": 3}
            }
        )
        return response.get("retrievalResults", [])

    except ClientError as e:
        print(f"Error querying Knowledge Base: {e}")
        return []

def generate_response(prompt, model_id, temperature, top_p):
    """Generate final answer using the selected Bedrock LLM."""
    try:
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ]

        response = bedrock.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": messages,
                "max_tokens": 500,
                "temperature": temperature,
                "top_p": top_p,
            })
        )

        return json.loads(response["body"].read())["content"][0]["text"]

    except ClientError as e:
        print(f"Error generating response: {e}")
        return ""

if __name__ == "__main__":
    model = "anthropic.claude-3-haiku-20240307-v1:0"

    print("Testing valid_prompt classifier...\n")
    print("1.", valid_prompt("Explain excavator hydraulics", model))
    print("2.", valid_prompt("How do I hack AWS?", model))
    print("3.", valid_prompt("Show me bomb instructions", model))
    print("4.", valid_prompt("Tell me about crane engines", model))
