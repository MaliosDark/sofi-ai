# SOFIA LLM Integration Setup Guide

## Overview
SOFIA now supports Large Language Model (LLM) integration for enhanced natural language generation and more sophisticated emotional responses. This guide explains how to set it up.

## Supported LLM Providers

### 1. OpenAI (Recommended)
- **Models**: GPT-3.5-turbo, GPT-4, GPT-4-turbo
- **Setup**:
  1. Get an API key from [OpenAI](https://platform.openai.com/api-keys)
  2. Set the environment variable: `export OPENAI_API_KEY="your-key-here"`
  3. Or edit `config.yaml` and add your key under `api_keys.openai`

### 2. Anthropic Claude
- **Models**: claude-3-sonnet-20240229, claude-3-haiku-20240307
- **Setup**:
  1. Get an API key from [Anthropic](https://console.anthropic.com/)
  2. Set: `export ANTHROPIC_API_KEY="your-key-here"`
  3. Or edit `config.yaml` under `api_keys.anthropic`

### 3. Groq (Fast & Free Tier)
- **Models**: llama2-70b-4096, mixtral-8x7b-32768
- **Setup**:
  1. Get an API key from [Groq](https://console.groq.com/)
  2. Set: `export GROQ_API_KEY="your-key-here"`
  3. Or edit `config.yaml` under `api_keys.groq`

### 4. Local Models (Free, runs on your machine)
- **Requirements**: transformers, torch
- **Models**: Any conversational model from HuggingFace
- **Setup**: Edit `config.yaml`:
  ```yaml
  llm_provider: "local"
  local_model_path: "microsoft/DialoGPT-medium"
  ```

## Configuration

Edit `config.yaml` to customize LLM behavior:

```yaml
llm_provider: "openai"  # openai, anthropic, groq, local
model_name: "gpt-3.5-turbo"
temperature: 0.7        # 0.0 = consistent, 1.0 = creative
max_tokens: 150         # Response length limit
api_keys:
  openai: "your-openai-key"
  anthropic: "your-anthropic-key"
  groq: "your-groq-key"
```

## Benefits of LLM Integration

1. **More Natural Responses**: Human-like conversation flow
2. **Better Context Understanding**: Deeper comprehension of user intent
3. **Enhanced Emotional Intelligence**: More nuanced emotional responses
4. **Complex Topic Handling**: Can discuss abstract concepts, stories, advice
5. **Adaptive Communication Style**: Learns and adapts to user preferences

## Fallback Behavior

If LLM is unavailable (no API key, network issues), SOFIA automatically falls back to its template-based responses, ensuring the system always works.

## Testing LLM Integration

Run the emotional AGI test suite:
```bash
cd /home/nexland/sofi-labs
source .venv/bin/activate
python test_sofia_emotional.py
```

Look for "🧠 LLM integration enabled" in the output. If you see "falling back to template responses", check your API key configuration.

## Performance Notes

- **OpenAI GPT-4**: Highest quality, slower, paid
- **Anthropic Claude**: Excellent quality, good speed, paid
- **Groq**: Fast inference, good quality, has free tier
- **Local Models**: Free, private, but requires good GPU for larger models

## Troubleshooting

1. **"LLM not available"**: Check API key environment variables
2. **Rate limits**: Implement delays between requests if needed
3. **Cost monitoring**: Track API usage for paid providers
4. **Model switching**: Change providers in config.yaml without code changes

## Advanced Configuration

For production use, consider:
- Response caching to reduce API calls
- Rate limiting to manage costs
- Model fine-tuning for domain-specific responses
- Custom prompts for specialized emotional support
