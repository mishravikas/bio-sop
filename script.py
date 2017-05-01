import Image
 
def get_average_color((x,y), n, image):
    """ Returns a 3-tuple containing the RGB value of the average color of the
    given square bounded area of length = n whose origin (top left corner) 
    is (x, y) in the given image"""
 
    r, g, b = 0, 0, 0
    count = 0
    for s in range(x, x+n+1):
        for t in range(y, y+n+1):
            pixlr, pixlg, pixlb = image[s, t]
            r += pixlr
            g += pixlg
            b += pixlb
            count += 1
    return ((r/count), (g/count), (b/count))
 
image = Image.open('im2.png').load()
r, g, b = get_average_color((0,0), 50, image)
print r,g,b