import pickle, numpy as np

a = pickle.load(open('known_faces_db.pkl', 'rb'))
b = pickle.load(open(r'workspace\gallery.pkl', 'rb'))

def arr(x):
    try:
        return np.asarray(x, dtype=float)
    except Exception:
        return None

print('hanya di lama:', sorted(set(a) - set(b)), '| hanya di app:', sorted(set(b) - set(a)))
for k in sorted(set(a) & set(b)):
    x, y = arr(a[k]), arr(b[k])
    if x is None or y is None:
        print(k, 'format tidak bisa dibandingkan:', type(a[k]).__name__, type(b[k]).__name__)
    elif x.shape != y.shape:
        print(k, 'BEDA bentuk', x.shape, y.shape)
    elif np.array_equal(x, y):
        print(k, 'sama persis', x.shape)
    else:
        print(k, 'BEDA nilai, selisih maks', round(float(np.abs(x - y).max()), 4), x.shape)