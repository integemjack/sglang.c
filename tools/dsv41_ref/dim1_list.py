import struct

VTS = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
for i in range(1, 6):
    path = f"./models/dsv41/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-{i:05d}-of-00006.gguf"
    fh = open(path, "rb")
    assert fh.read(4) == b"GGUF"
    struct.unpack("<I", fh.read(4))
    n_ten, n_kv = struct.unpack("<QQ", fh.read(16))

    def rstr():
        n = struct.unpack("<Q", fh.read(8))[0]
        return fh.read(n)

    for _ in range(n_kv):
        rstr()
        t = struct.unpack("<I", fh.read(4))[0]
        if t == 8:
            rstr()
        elif t == 9:
            et = struct.unpack("<I", fh.read(4))[0]
            n = struct.unpack("<Q", fh.read(8))[0]
            if et == 8:
                for _ in range(n):
                    rstr()
            else:
                fh.seek(n * VTS[et], 1)
        else:
            fh.seek(VTS[t], 1)
    for _ in range(n_ten):
        name = rstr().decode()
        nd = struct.unpack("<I", fh.read(4))[0]
        ne = struct.unpack("<" + "Q" * nd, fh.read(8 * nd))
        gtype = struct.unpack("<I", fh.read(4))[0]
        off = struct.unpack("<Q", fh.read(8))[0]
        if nd == 1 and "_exps" not in name and not name.startswith("engram") and "per_layer" not in name:
            print(f"1-D: {name} ne={ne} gtype={gtype}")
