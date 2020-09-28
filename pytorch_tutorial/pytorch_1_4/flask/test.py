import requests

resp = requests.post("http://localhost:5000/predict",
                     files={"file": open('/Users/binyu/Documents/git_exercise/pytorch_tutorial/pytorch_1_4/data/faces/252418361_440b75751b.jpg', 'rb')})
print(resp.json())

