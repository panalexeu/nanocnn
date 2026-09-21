from data import download_usps, get_usps

if __name__ == '__main__': 
    download_usps()
    ds = get_usps() 
    print(ds.keys())
    print(ds['train'])
    print(ds['test'])
