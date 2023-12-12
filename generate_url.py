def generate_url_list(input_file, output_file):
    base_url = "https://danbooru.donmai.us/posts?tags={}_%28arknights%29+solo&z=5"

    with open(input_file, 'r') as file:
        names = file.readlines()

    urls = [base_url.format(name.strip().replace(' ', '_').replace('(', '%28').replace(')', '%29').lower()) for name in names]

    with open(output_file, 'w') as file:
        for url in urls:
            file.write(url + '\n')

generate_url_list('input.txt', 'output.txt')
