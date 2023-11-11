# python3
'''
    train model
    input is a scaler and a model
    input is also training data
'''

for ndx, vect in enumerate(X_data):
    if len(vect) != 29:
        print(ndx, "bad length", len(vect), vect)
    for n in vect:
        try:
            f = float(n)
        except Exception as ex:
            print(ndx, "bad number", vect)

scaler.fit(X_data)
X_data = scaler.transform(X_data)

epochs, batchsz = 10, None
if 'epochs' in parmdict:
    epochs = parmdict['epochs']
if 'batchsz' in parmdict:
    batchsz = parmdict['batchsz']

try:
    history = model.fit(
        X_data, y_classes,
        epochs=epochs, batch_size=batchsz, verbose=0)
except ValueError as excp:
    print("keras model fit error", excp)
    print("y_classes", len(y_classes), y_classes)
    print("X_data", type(X_data), X_data.shape)
    raise
# plot_train_history(history, model_id)

# save trained model to disk  for keras, this is more complicated
# serialize model to HDF5 format
model_fn = str(cfg.mlmod / model_id)
model.save(model_fn + '.h5')
# The scaler must also be saved
pdata = dict(model=None, scaler=scaler)
with open(model_fn + '.pickle', "wb") as pf:
    pickle.dump(pdata, pf)
