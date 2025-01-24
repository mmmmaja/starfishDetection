# Application

Users can interact with our application using the backend or the frontend.

## Backend
To obtain predictions on an image by communicating with the API directly, you can use a curl command.
```bash
curl -X 'POST' 'https://backend-638730968773.us-central1.run.app/inference/' -H 'accept: application/json' -H 'Content-Type: multipart/form-data' -F 'data=@PATH_TO_IMAGE;type=image/jpeg'
```

## Frontend
For a more user-friendly experience, you can access our frontend webpage at [https://frontend-638730968773.us-central1.run.app](https://frontend-638730968773.us-central1.run.app).

Follow these steps to get predictions on an image.

1. Click `Browse files`.

2. Select your desired image. You should then see the image you selected appear.

3. 
