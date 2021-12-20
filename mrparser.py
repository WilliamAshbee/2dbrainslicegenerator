fn = 'rh.white.txt'
import os

count = 0 

with open('rh.white.txt') as infile:
    id4Prev = None
    mri = None
    surface = None
    while True:
        currentPath = infile.readline()
        path = os.path.normpath(currentPath)
        patharr = path.split(os.sep)

        #print(currentPath,type(currentPath))
        count+=1
        
        try:
            id4 = patharr[4]
            id6 = patharr[6]
            cfname = patharr[-1]
            assert id4 == id6
            
            if id4!= id4Prev:
                assert mri == None
                assert surface == None
                id4Prev = id4
            
            if 'touch' in currentPath:
                continue
            elif 'deformed' in currentPath:
                continue
            elif 'hires' in currentPath:
                continue
            elif 'nii.gz' in currentPath:
                print('lastpath',patharr[-1].strip())
                mri = True
            elif patharr[-1].strip() == 'rh.white':
                print('lastpath',patharr[-1].strip())
                surface = True
            else:
                continue
            
            if mri != None and surface !=None:
                assert id4 == id4Prev #assert every file has both an mri and surface
                print('create input output pair')
                mri = None
                surface = None    
            
            print(id4,id6)
            
            print(count)
            print()
        except:
            assert currentPath == ''
            break
        