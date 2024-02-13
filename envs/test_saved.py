from tensorflow import keras
# Recreate the exact same model purely from the file
new_model = keras.models.load_model('drl_model.h5')