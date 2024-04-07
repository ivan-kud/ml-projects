from fastapi import FastAPI, Request, Form, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import digit
from . import instance
from . import sentiment
from . import search
from . import suggestion
from . import housing


app = FastAPI()

templates = Jinja2Templates(directory='templates')
app.mount('/static', StaticFiles(directory='static'), name='static')


@app.get('/', response_class=HTMLResponse)
async def read_home(request: Request):
    return templates.TemplateResponse('pages/index.html', {'request': request})


@app.get('/digit', response_class=HTMLResponse)
async def read_digit(request: Request):
    return templates.TemplateResponse('pages/digit.html', {'request': request})


@app.post('/digit', response_class=HTMLResponse)
async def predict_digit(request: Request,
                        model: digit.ModelName = Form(),
                        image: str = Form()):
    response = digit.get_response(model, image)
    response.update({'request': request})
    return templates.TemplateResponse('pages/digit.html', response)


@app.get('/instance', response_class=HTMLResponse)
async def read_instance(request: Request):
    return templates.TemplateResponse('pages/instance.html',
                                      {'request': request})


@app.post('/instance', response_class=HTMLResponse)
async def predict_instance(request: Request,
                           file: UploadFile):
    response = instance.get_response(file)
    response.update({'request': request})
    return templates.TemplateResponse('pages/instance.html', response)


@app.get('/sentiment', response_class=HTMLResponse)
async def read_sentiment(request: Request):
    return templates.TemplateResponse('pages/sentiment.html',
                                      {'request': request})


@app.post('/sentiment', response_class=HTMLResponse)
async def predict_sentiment(request: Request,
                            text: str = Form()):
    response = sentiment.get_response(text)
    response.update({'request': request})
    return templates.TemplateResponse('pages/sentiment.html', response)


@app.get('/search', response_class=HTMLResponse)
async def read_search(request: Request):
    return templates.TemplateResponse('pages/search.html',
                                      {'request': request})


@app.post('/search', response_class=HTMLResponse)
async def predict_search(request: Request,
                         file: UploadFile,
                         model: search.ModelName = Form()):
    response = search.get_response(model, file)
    response.update({'request': request})
    return templates.TemplateResponse('pages/search.html', response)


@app.get('/suggestion', response_class=HTMLResponse)
async def read_suggestion(request: Request):
    return templates.TemplateResponse('pages/suggestion.html',
                                      {'request': request})


@app.post('/suggestion', response_class=HTMLResponse)
async def predict_suggestion(request: Request,
                             text: str = Form()):
    response = suggestion.get_response(text)
    response.update({'request': request})
    return templates.TemplateResponse('pages/suggestion.html', response)


@app.post('/housing')
async def predict_housing(housings: list[housing.Housing]):
    try:
        result = housing.predict(housings)
    except Exception:
        return None
    return result
