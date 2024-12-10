# Comfy UIAPI

UIAPI is an intermediate and frontend plugin which allow communicating with the Comfy webui through server connection. This saves the need to export a workflow.json and instead directly sending a queue command to the frontend. This way, the user can experiment in realtime as they are running some professional industry or rendering software which uses UIAPI / ComfyUI as a backend. There is no way to switch seamlessly between UIAPI and regular server connection - though as of late summer 2023 it was inferior to use the server connection because the server would constantly unload models and start from scratch, and the schema of the workfow json was completely different and much less convenient, losing crucial information for efficient querying of nodes and assigning data dynamically.

## How?

1. Both the frontend and your program connect to Comfy through websockets.
2. The UIAPI plugin implements new API routes.
3. Your program invokes the special API routes.
4. The UIAPI plugin acts as a repeater and forwards the requests to the UIAPI javascript plugin running in the webui page.
5. The request is handled, and a response is posted
6. The UIAPI acts as a repeater again, this time forwarding to all the other connections
7. Your program handles the response and unlocks program execution

## Support Calls

* `comfy.connect(path1, path2)` between inputs and outputs of nodes
* `comfy.set(path, value)` Set values on widgets (numbers and strings)
* `comfy.execute()` Queue / execute workflow and receive the output image (detected on a list of common Save Image components)

## Downloads

Since it's typical to run ComfyUI on a server and connect to it locally, routing features are provided to download the missing models. To use it,

```
TODO
```
