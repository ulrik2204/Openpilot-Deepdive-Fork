def main():
    print("Hello World!")
    with open("test.txt", "w") as f:
        f.write("Hello World! from file")


if __name__ == "__main__":
    main()
