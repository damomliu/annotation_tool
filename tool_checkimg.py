import os
from tqdm import tqdm
from annotation.app import LabelImgXML, LabelmeJSON
from annotation.image import ImageFile

FOLDER = '/media/alpha/liebe/vreu_data/newdata_0618/images/from114'
if __name__=='__main__':
    counts = dict(json=0, xml=0, image=0, others=0, unmatch=0)
    pbar = tqdm(total=sum([len(files) for _,_,files in os.walk(FOLDER)]))
    for root,_,files in os.walk(FOLDER):
        for f in files:
            fname,ext = os.path.splitext(f)
            fp = os.path.join(root,f)
            
            ext = ext.lower()
            try:
                if ext=='.json':
                    jf = LabelmeJSON(fp)
                    if jf.imgfile.w:
                        counts['json'] += 1
                elif ext=='.xml':
                    xml = LabelImgXML(fp)
                    if xml.imgfile.w:
                        counts['xml'] += 1
                elif ext in ['.jpg','.jpeg','.png']:
                    imgf = ImageFile(fp)
                    if imgf.w:
                        counts['image'] += 1
                else:
                    print(f'uncategorized filetype [{fp}]')
                    counts['others'] += 1
            except:
                print(f'error occurs in [{fp}]')
                counts['unmatch'] += 1
            
            pbar.update()
            pbar.set_postfix(counts)
    pbar.close()