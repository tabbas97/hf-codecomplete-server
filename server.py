import argparse
import json
from typing import AsyncGenerator
import uvicorn

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()
app.engine = None

@app.post("/api/generate/{modelname:path}")
async def generate(request : Request) -> Response:

    json_dict = await request.json()

    # Parameters need to be popped. If not popped, affects later initialization calls
    prompt = json_dict.pop("inputs")
    params = json_dict.pop("parameters")
    streaming_mode = json_dict.pop("stream", False)
    max_nt = params.pop("max_new_tokens")
    ret_ft = params.pop("return_full_text", False)
    sample = params.pop("do_sample", True)

    # We use a random UUID to identify the request
    req_id = random_uuid()

    # When attempting to code, the extension tends to throw a large number
    # of requests, which the server will likely not be able to handle. 
    # When a request is aborted on the client side, the server will still
    # attempt to process it, and will waste resources.
    # We will cancel the request if it is aborted from the client side

    results_gen = app.engine.generate(
        prompt,
        SamplingParams(
            max_tokens=max_nt,
            use_beam_search=sample,
            **params
        ),
        req_id
    )

    if streaming_mode:
        async def streaming_result() -> AsyncGenerator[str, None]:
            async for result in results_gen:
                prompt = result.prompt
                text_out = []
                for output in result.outputs:
                    text_out.append(
                        prompt + output.text if ret_ft else output.text
                        )
                    
                yield (json.dumps({
                    "text": text_out,
                }) + "\0").encode("utf-8")

        return StreamingResponse(
            streaming_result()
        )
    
    else:
        output = None
        async for result in results_gen:
            if await request.is_disconnected():
                await app.engine.abort(req_id)
                return JSONResponse(
                    {"error": "Request aborted"},
                    status_code=499
                )
            output = result

        if output is not None:
            prompt = output.prompt
            text_out = []
            for output in output.outputs:
                text_out.append(
                    prompt + output.text if ret_ft else output.text
                    )
            return JSONResponse({
                "generated_text": text_out[0],
                "status":200
            })
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    logging.info(f"Starting server on {args.host}:{args.port}")
    
    app.engine = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs.from_cli_args(
            args
        )
    )

    uvicorn.run(
        app, 
        host=args.host, 
        port=args.port,
        log_level="info"
        )