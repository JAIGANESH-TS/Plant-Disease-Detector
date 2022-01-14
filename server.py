import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

export_file_url = 'https://drive.google.com/uc?export=download&id=1qGyr9AEj71iLITNpjBaKqY7DYJ2xrVfm'
export_file_name = 'export_resnet34_model.pkl'
export_file_path = Path(__file__).parent / 'models'

classes = ['Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy','Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy','Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_','Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy','Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy','Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy','Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight','Potato___Late_blight','Potato___healthy','Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy','Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus','Tomato___healthy','background']
diseaseName = {
    'Apple___Apple_scab':'Materials available to home growers for scab control in edible apples and crabapples include captan, lime-sulfur and powdered or wettable sulfur. Applications of lime-sulfur closely following captan sprays can damage leaves and flower buds, so use caution when rotating these two materials',
    'Apple___Black_rot':'The best way to control black rot is to keep trees in a good state of vigor through annual pruning, watering during summer droughts, and other beneficial cultural practices. Devitalized trees develop more twig,limb, and trunk cankers than vigorous ones. ',
    'Apple___Cedar_apple_rust':'Fungicides with the active ingredient Myclobutanil are most effective in preventing rust.Fungicides are only effective if applied before leaf spots or fruit infection appear.Spray trees and shrubs when flower buds first emerge until spring weather becomes consistently warm and dry.Monitor nearby junipers.',
    'Apple___healthy':'There is no disease on the Apple leaf',
    'Blueberry___healthy':'There is no disease on the Blueberry leaf',
    'Cherry_(including_sour)___Powdery_mildew':'The key to managing powdery mildew on the fruit is to keep the disease off of the leaves. Most synthetic fungicides are preventative, not eradicative, so be pro-active about disease prevention. Maintain a consistent program from shuck fall through harvest.',
    'Cherry_(including_sour)___healthy':'There is no disease on the Cherry leaf',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot':'Disease management tactics include using resistant corn hybrids, conventional tillage where appropriate, and crop rotation. Foliar fungicides can be effective if economically warranted.',
    'Corn_(maize)___Common_rust_':'The best management practice is to use resistant corn hybrids. Fungicides can also be beneficial, especially if applied early when few pustules have appeared on the leaves.',
    'Corn_(maize)___Northern_Leaf_Blight':'Treating northern corn leaf blight involves using fungicides. For most home gardeners this step isnt needed, but if you have a bad infection, you may want to try this chemical treatment. The infection usually begins around the time of silking, and this is when the fungicide should be applied.',
    'Corn_(maize)___healthy':'There is no disease on the Corn leaf',
    'Grape___Black_rot':'Careful handling and prompt refrigeration to 1-2 C or below prevents the disease in storage. Inclusion of SO releasing pads in the 2 boxes while packing helps to control the disease.',
    'Grape___Esca_(Black_Measles)':'Mummified berries left on vines should be collected and destroyed. Cultivation practices should ensure free circulation of air. Spraying Bordeaux mixture (4:4:100) once or twice on young bunches prevents the infection. Copper fungicides are preferred for spraying on bunches, as they do not leave any visible deposits on the fruit surface.',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)':'If the disease on the berries is not controlled in the field, it can lead to berry rotting during transit and storage. Bordeaux mixture (1.0%), Mancozeb (0.2%), Topsin-M (0.1%), Ziram (0.35%) orCaptan (0.2%) is to be sprayed alternatively at weekly intervals from Jun-August and again from December until harvest to keep this disease under check. Two to three sprays of systemic fungicides should be given per season.',
    'Grape___healthy':'There is no disease on the Grape leaf',
    'Orange___Haunglongbing_(Citrus_greening)':'Once a tree has citrus greening, there is no cure. Over time, your tree will deteriorate and the disease will ultimately destroy the tree. It is incredibly important to remove trees that have citrus greening disease.',
    'Peach___Bacterial_spot':'Compounds available for use on peach and nectarine for bacterial spot include copper, oxytetracycline (Mycoshield and generic equivalents), and syllit+captan; however, repeated applications are typically necessary for even minimal disease control.',
    'Peach___healthy':'There is no disease on the Peach leaf',
    'Pepper,_bell___Bacterial_spot':'Seed treatment with hot water, soaking seeds for 30 minutes in water pre-heated to 125 F/51 C, is effective in reducing bacterial populations on the surface and inside the seeds. However, seed germination may be affected by heat treatment if not done accurately, while the risk is relatively low with bleach treatment.',
    'Pepper,_bell___healthy':'There is no disease on the Pepperbell leaf',
    'Potato___Early_blight':'Treatment of early blight includes prevention by planting potato varieties that are resistant to the disease; late maturing are more resistant than early maturing varieties. Avoid overhead irrigation and allow for sufficient aeration between plants to allow the foliage to dry as quickly as possible',
    'Potato___Late_blight':'The severe late blight can be effectively managed with prophylactic spray of mancozeb at 0.25 Percentage followed by cymoxanil+mancozeb or dimethomorph+mancozeb at 0.3 Percentage at the onset of disease and one more spray of mancozeb at 0.25 Percentage seven days after application of systemic fungicides in West Bengal [50]',
    'Potato___healthy':'There is no disease on the Potato leaf',
    'Raspberry___healthy':'There is no disease on the Raspberry leaf',
    'Soybean___healthy':'There is no disease on the Soybean leaf',
    'Squash___Powdery_mildew':'Use resistant varieties when they are available. Thin plants to proper spacing so each leaf gets good exposure to sun and fresh air. Plant fast-growing varieties of summer squash to sidestep this disease. Starting in early summer, spray plants every 10 days with a mixture of one part milk (any kind) to four parts water.',
    'Strawberry___Leaf_scorch':'These practices include the use of proper plant spacing to provide adequate air circulation and the use of drip irrigation. The avoidance of waterlogged soil and frequent garden cleanup will help to reduce the likelihood of spread of this fungus.',
    'Strawberry___healthy':'There is no disease on the Strawberry leaf',
    'Tomato___Bacterial_spot':'A plant with bacterial spot cannot be cured.  Remove symptomatic plants from the field or greenhouse to prevent the spread of bacteria to healthy plants.  Burn, bury or hot compost the affected plants and DO NOT eat symptomatic fruit.  Although bacterial spot pathogens are not human pathogens, the fruit blemishes that they cause can provide entry points for human pathogens that could cause illness.',
    'Tomato___Early_blight':'Tomatoes that have early blight require immediate attention before the disease takes over the plants. Thoroughly spray the plant (bottoms of leaves also) with Bonide Liquid Copper Fungicide concentrate or Bonide Tomato & Vegetable. Both of these treatments are organic.',
    'Tomato___Late_blight':'Tomatoes that have early blight require immediate attention before the disease takes over the plants. Thoroughly spray the plant (bottoms of leaves also) with Bonide Liquid Copper Fungicide concentrate or Bonide Tomato & Vegetable. Both of these treatments are organic..',
    'Tomato___Leaf_Mold':'Use drip irrigation and avoid watering foliage. Use a stake, strings, or prune the plant to keep it upstanding and increase airflow in and around it. Remove and destroy (burn) all plants debris after the harvest.',
    'Tomato___Septoria_leaf_spot':'Removing infected leaves: Remove infected leaves immediately, and be sure to wash your hands and pruners thoroughly before working with uninfected plants. Consider organic fungicide options: Fungicides containing either copper or potassium bicarbonate will help prevent the spreading of the disease. Begin spraying as soon as the first symptoms appear and follow the label directions for continued management. Consider chemical fungicides: While chemical options are not ideal, they may be the only option for controlling advanced infections. One of the least toxic and most effective is chlorothalonil (sold under the names Fungonil and Daconil).',
    'Tomato___Spider_mites Two-spotted_spider_mite':'abc',
    'Tomato___Target_Spot':'Many fungicides are registered to control of target spot on tomatoes. Growers should consult regional disease management guides for recommended products. Products containing chlorothalonil, mancozeb, and copper oxychloride have been shown to provide good control of target spot in research trials',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus':'Inspect plants for whitefly infestations two times per week. If whiteflies are beginning to appear, spray with azadirachtin (Neem), pyrethrin or insecticidal soap. For more effective control, it is recommended that at least two of the above insecticides be rotated at each spraying.',
    'Tomato___Tomato_mosaic_virus':'There are no cures for viral diseases such as mosaic once a plant is infected. As a output, every effort should be made to prevent the disease from entering your garden.1.Fungicides will NOT treat this viral disease.2.Plant resistant varieties when available or purchase transplants from a reputable source.3.Do NOT save seed from infected crops.4.Spot treat with least-toxic, natural pest control products, such as Safer Soap, Bon-Neem and diatomaceous earth, to reduce the number of disease carrying insects.5.Harvest-Guard® row cover will help keep insect pests off vulnerable crops/ transplants and should be installed until bloom.6.Remove all perennial weeds, using least-toxic herbicides, within 100 yards of your garden plot.7.The virus can be spread through human activity, tools and equipment. Frequently wash your hands and disinfect garden tools, stakes, ties, pots, greenhouse benches, etc. (one part bleach to 4 parts water) to reduce the risk of contamination.8.Avoid working in the garden during damp conditions (viruses are easily spread when plants are wet).9.Avoid using tobacco around susceptible plants. Cigarettes and other tobacco products may be infected and can spread the virus.10.Remove and destroy all infected plants (see Fall Garden Cleanup).',
    'Tomato___healthy':'There is no disease on the Tomato leaf.'
    }

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists():
        return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, export_file_path / export_file_name)
    try:
        learn = load_learner(export_file_path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

index_path = Path(__file__).parent

@app.route('/')
async def homepage(request):
    index_file = index_path / 'view' / 'index.html'
    return HTMLResponse(index_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': '<h6>' + str(prediction) + '</h6></br><p>' + 'Solution:' + diseaseName[str(prediction)] + '</p>'})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=8080, log_level="info")
