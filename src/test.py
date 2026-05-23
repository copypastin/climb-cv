from climbcv.climbcv import climbcv

def main():
    ccv = climbcv()
    ccv.start()

    while True:
        print(ccv.raw_landmarks)

if __name__ == "__main__":
    main()