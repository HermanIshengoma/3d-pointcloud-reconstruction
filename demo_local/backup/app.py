# app.py
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from sdf_model import call_model1, call_model3 #, call_model
import numpy as np
import ast
#from plyfile import PlyData, PlyElement
app = Flask(__name__)
CORS(app)

@app.route('/retrievemeshes/', methods=['POST'])
def respond():
    # # Retrieve the name from url parameter

    # name = request.args.get("name", None)

    # # For debugging
    # print(f"got name {name}")

    # response = {}

    # # Check if user sent a name at all
    # if not name:
    #     response["ERROR"] = "no name found, please send a name."
    # # Check if the user entered a number not a name
    # elif str(name).isdigit():
    #     response["ERROR"] = "name can't be numeric."
    # # Now the user entered a valid name
    # else:
    #     response["MESSAGE"] = f"Welcome {name} to our awesome platform!!"

    # # Return the response in json format
    # #return jsonify(response)
    # return jsonify("Your request has been received")
    if request.method == 'POST':
        #meshID =request.get_json()['meshID'] # integer
        return send_file('sdf_model/input/reconstruct.ply', attachment_filename='mesh.ply')

@app.route('/post/', methods=['POST'])
def receive_point_clouds():
    if request.method == 'POST':
        # Get request parameters
        # get point clouds

        #print("data: ", request.get_json()['data'])
        #print("resolution: ", request.get_json()['resolution'], type(request.get_json()['resolution'])) #integer
        #print("type: ", request.get_json()['type'], type(request.get_json()['type']))


        point_clouds = np.array(ast.literal_eval(request.get_json()['data']))
        resolution = request.get_json()['resolution']
        model_type = request.get_json()['type']



        # prepare response
        print(type(point_clouds),point_clouds.shape)
        call_sdf_model(model_type, resolution, point_clouds) 
        return send_file('sdf_model/output/reconstruct.ply', attachment_filename='mesh.ply')


# A welcome message to test our server
@app.route('/')
def index():
    return "<h1>Welcome to our server !!</h1>"


# machine learning function
def call_sdf_model(model_type, resolution, pc):
	
    if model_type == 'Level1':
        print("model1 called")
        cd = call_model1.main(resolution, None)
    elif model_type == 'Level3':
        print("model3 called")
        cd = call_model3.main(resolution, pc)

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)