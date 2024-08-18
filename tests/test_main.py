# from fastapi.testclient import TestClient

# from app.main import app

# client = TestClient(app)


# def test_create_vector():
#     response = client.post("/vectors/", json={"text": "Hello World", "metadata": {}})
#     assert response.status_code == 200
#     assert "id" in response.json()


# def test_get_vector_not_found():
#     response = client.get("/vectors/nonexistent-id")
#     assert response.status_code == 404


# def test_search_vectors():
#     response = client.post("/search/", json={"query": "Hello", "top_k": 5})
#     assert response.status_code == 200
#     assert isinstance(response.json(), list)
