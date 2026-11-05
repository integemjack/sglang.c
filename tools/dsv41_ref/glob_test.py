import glob

p = "/Volumes/9100 PRO/models/dsv41/"
c = sorted(glob.glob(p + "DeepSeek*IQ2_XXS*.gguf"))
print(len(c))
for x in c[:6]:
    print(repr(x))
    try:
        f = open(x, "rb")
        print("  open ok", f.read(4))
        f.close()
    except OSError as e:
        print("  ERR", e)