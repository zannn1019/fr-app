import cv2
from insightface.app import FaceAnalysis

app = FaceAnalysis(
    providers=["CPUExecutionProvider"],
)

app.prepare(
    ctx_id=0,
    det_size=(800, 800)
)

camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()    
    frame = cv2.flip(frame, 1)
    if not success:
        break

    faces = app.get(frame)

    for face in faces:
        score = face.det_score

        x1, x2, y1, y2 = face.bbox.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Score: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()