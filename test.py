import time

from tqdm import tqdm


def main():
    print("Hello World!")
    with open("test.txt", "w") as f:
        f.write("Hello World! from file")
    for _ in tqdm(range(10)):
        time.sleep(2)
    for j in range(10):
        if j % 2 == 0:
            print("index", j)
        time.sleep(2)


if __name__ == "__main__":
    main()
