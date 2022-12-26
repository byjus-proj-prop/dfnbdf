# Para capturar os quadros
import cv2

# Para processar o array de imagens
import numpy as np


# importe os módulos tensorflow e carregue o modelo
import tensorflow as tf

model = tf.keras.models.load_model("keras_model.h5");


# Anexando a câmera indexada como 0, com o software da aplicação
camera = cv2.VideoCapture(0)

# Loop infinito
while True:

	# Lendo / requisitando um quadro da câmera 
	status , frame = camera.read()

	# Se tivemos sucesso ao ler o quadro
	if status:

		# Inverta o quadro
		frame = cv2.flip(frame , 1)
		
		
		sucess, frame = camera.read();img = cv2.resize(frame, (224,224));test_img = np.array(img, dtype=np.float32);test_img = np.expand_dims(test_img, axis=0);img_normal = test_img/255.0;prediction = model.predict(img_normal);print("previsão: ", prediction);cv2.imshow("frame", frame); 
        if cv2.waitKey(1) == 32:print("fechando");break
        
		
		
		# Exibindo os quadros capturados
		cv2.imshow('feed' , frame)

		# Aguardando 1ms
		code = cv2.waitKey(1)
		
		# Se a barra de espaço foi pressionada, interrompa o loop
		if code == 32:
			break

# Libere a câmera do software da aplicação
camera.release()

# Feche a janela aberta
cv2.destroyAllWindows()
