import time

from tqdm import tqdm


def main():
    print("Hello World!")
    with open("test.txt", "w") as f:
        f.write("Hello World! from file")
    for i in tqdm(range(10)):
        print("index", i)
        time.sleep(2)


if __name__ == "__main__":
    main()
