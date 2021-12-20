from fastapi.testclient import TestClient
from app.main import app

"""
These tests are very bad. However, something is better than nothing.
"""

client = TestClient(app)


def test_get_model_list():
    response = client.get("/api/v1/models")
    assert response.status_code == 200


# This assumes that the database table is ready and there is a user with username "test" and password "test"!
def test_login():
    response = client.post("/api/v1/users/token",
                           json={"username": "test", "password": "test"}
                           )
    assert response.status_code == 200
    assert response.json()["access_token"] is not None
    assert response.json()["token_type"] == "bearer"


# It might be a good idea to make use of the application config for testing endpoints requiring auth.
def test_login_load_infer():
    login_response = client.post("/api/v1/users/token",
                                 json={"username": "test", "password": "test"}
                                 )
    assert login_response.status_code == 200
    token = login_response.json()["access_token"]

    test_model = "distilgpt2"
    load_model_response = client.post("/api/v1/models/load",
                                      headers={"Authorization": f"Bearer {token}"},
                                      json={"model": f"{test_model}", "parallel": "false", "sharded": "false"}
                                      )
    assert load_model_response.status_code == 200

    infer_model_response = client.post("/api/v1/models/generate",
                                       headers={"Authorization": f"Bearer {token}"},
                                       json={
                                            "model": f"{test_model}",
                                            "prompt": "Hello! My name is",
                                            "sample_args": {
                                              "temp": 0.51,
                                              "top_p": 0.9,
                                              "top_k": 140,
                                              "tfs": 0.993,
                                              "rep_p": 1.3,
                                              "rep_p_range": 1024,
                                              "rep_p_slope": 0.18,
                                              "bad_words": [
                                                "Jack"
                                              ],
                                              "bias_words": [
                                                "Melissa"
                                              ],
                                              "bias": 5.0
                                            },
                                            "gen_args": {
                                              "max_length": 10
                                            }
                                          }
                                       )
    assert infer_model_response.status_code == 200
    assert infer_model_response.json()["completion"] is not None
    assert infer_model_response.json()["completion"]["text"] is not None
    assert infer_model_response.json()["completion"]["time"] > 0
