import asyncio
from sofia_llm_integration import SOFIALanguageModel

async def test():
    llm = SOFIALanguageModel()
    response = await llm.generate_response('Hello, how are you?')
    print("Response:", response)

if __name__ == "__main__":
    asyncio.run(test())
