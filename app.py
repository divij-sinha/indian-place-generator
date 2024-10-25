from fastapi import FastAPI, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from utils import generate_word_from_save, state_code, state_code_rev

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
@app.post("/")
async def root(
    request: Request,
    start_text: str = Form(default=""),
    n: int = Form(default=1),
    states: list[str] = Form(default=[]),
):
    generations = []
    if request.method == "POST":
        print(states)
        states = [state_code[s] for s in states if s != ""]
        print(states)
        generations = generate_word_from_save(start_text.lower(), n, states)
    fixed_generations = []
    for g in generations:
        state, word = g.split("?")
        if state != "":
            state = state_code_rev[int(state.replace("~", ""))]
        word = word.replace("!", "").title()
        fixed_generations.append((word, state))
    print(f"{start_text=}")
    print(f"{n=}")
    print(f"{states=}")
    return templates.TemplateResponse(
        "index.html", {"request": request, "generations": fixed_generations}
    )


# @app.post("/generate_word")
