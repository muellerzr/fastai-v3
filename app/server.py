from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai import *
from fastai.text import *
from fastai.tabular import *


model_file_url = 'https://www.dropbox.com/s/i5mbivnaocbshqz/good_model_epoc_3.pth?dl=0'
model_file_name = 'goodModel'

data_clas_url = 'https://www.dropbox.com/s/wo8f9xoqnxe9ag7/data_clas_export.pkl?dl=1'
data_class_name = 'data_clas_export'

#Testing




path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(data_clas_url, path/f'{data_class_name}.pkl')
    print(path/f'{data_class_name}.pkl')
    await download_file(model_file_url, path/'models'/f'{model_file_name}.pth')
    print(path/'models'/f'{model_file_name}.pth')

    data_clas = load_data(path, 'data_clas_export.pkl')

    learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5);
    learn.load(model_file_name)
     
    return learn

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    content = data['content']
    prediction = learn.predict(content)[2]
    reliability = prediction[7]-(prediction[6]*prediction[0]) - prediction[0] - prediction[4] - prediction[5] - prediction[8] - (prediction[1]-prediction[11])
    return JSONResponse({'result': str(reliability)})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app, host='0.0.0.0', port=5042)

