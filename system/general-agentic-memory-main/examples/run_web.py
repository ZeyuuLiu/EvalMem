#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run GAM Web Platform

Start the GAM web interface for building, managing and browsing GAM.
"""

import argparse

def main():
    parser = argparse.ArgumentParser(description='Run GAM Web Platform')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='',
        help='Pipeline output root directory (default: built-in path in app.py)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port to bind to (default: 5000)'
    )
    parser.add_argument(
        '--no-llm',
        action='store_true',
        help='Disable LLM (content will be stored as-is)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model name for LLM (Env: GAM_MODEL, default: gpt-4o-mini)'
    )
    parser.add_argument(
        '--api-base',
        type=str,
        default=None,
        help='API base URL for LLM (Env: GAM_API_BASE, default: https://api.openai.com/v1)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='API key for LLM (Env: GAM_API_KEY)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=40960,
        help='Maximum tokens for LLM (default: 4096)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.3,
        help='Temperature for LLM (default: 0.3)'
    )
    parser.add_argument(
        '--no-debug',
        action='store_true',
        help='Disable debug mode'
    )
    
    args = parser.parse_args()
    
    import os

    generator = None
    if not args.no_llm:
        api_key = args.api_key or os.environ.get("GAM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("⚠️  No API key provided (use --api-key or set GAM_API_KEY)")
            print("   Continuing without LLM - content will be stored as-is")
        else:
            try:
                from gam import OpenAIGenerator, OpenAIGeneratorConfig

                model = args.model or os.environ.get("GAM_MODEL", "gpt-4o-mini")
                api_base = args.api_base or os.environ.get("GAM_API_BASE", "https://api.openai.com/v1")
                config = OpenAIGeneratorConfig(
                    model_name=model,
                    base_url=api_base,
                    api_key=api_key,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )

                print(f"🔗 Connecting to LLM: {model} at {api_base}")
                generator = OpenAIGenerator(config)
                print(f"✅ LLM connected successfully!")

            except Exception as e:
                print(f"⚠️  Could not connect to LLM: {e}")
                print(f"   Continuing without LLM - content will be stored as-is")
                generator = None
    
    # Run server
    from gam.web import run_server
    
    run_server(
        generator=generator,
        output_base=args.output_dir,
        host=args.host,
        port=args.port,
        debug=not args.no_debug
    )


if __name__ == '__main__':
    main()
