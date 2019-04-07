from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai import *
from fastai.text import *
from fastai.tabular import *

model_file_url = 'https://www.dropbox.com/s/x6p73q0kwei7mea/goodModel.pkl?dl=1'
model_file_name = 'export'


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
    await download_file(model_file_url, path/'models'/f'{model_file_name}.pth')
    #await download_file(data_clas_url, path/'models'/f'{data_class_name}.pth')

    learn= load_learner(path/'models')
     
    return learn

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
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
    #reliability = int(reliability)
    EddyScore = ((reliability.item())*50)+50
    #print(EddyScore)
    EddyScore = int(EddyScore)
    #if EddyScore < 40:
    #    return JSONResponse({'Reliability score': str(EddyScore),'Recommendation:':'This source is not reliable in the slightest. We recommend searching other avenues for news and current events.'})
        #return JSONResponse(data)
    #elif (EddyScore < 60) and (EddyScore > 40):
    #    return JSONResponse({'Reliability score': str(EddyScore),'Recommendation:':'This source is decently unreliable, check into any strong claims that have been made.'})
        #return JSONResponse(data)
    #elif (EddyScore < 80) and (EddyScore > 60):
    #    return JSONResponse({'Reliability score': str(EddyScore),'Recommendation:':'This source is fairly reliable, minimum research is recommended into claims.'})
        #return JSONResponse(data)
    #else:
     #   return JSONResponse({'Reliability score': str(EddyScore),'Recommendation:':'This source is extremely reliable and gets the Suspecto Seal of Approval!'})
        #return JSONResponse(data)
    #return JSONResponse({'result': str(reliability)})
    return JSONResponse({'result': EddyScore})
   

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app, host='0.0.0.0', port=5042)


