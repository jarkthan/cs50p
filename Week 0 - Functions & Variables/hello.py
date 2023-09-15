# Ask user for their name
#name = input("What's your name? ")

# Remove whitespace from str
#name = name.strip()

# Capitalize name
#name = name.capitalize()

# Title the name (better than capitalize in this case)
#name = name.title()

##Remove whitespace and capitalize user's name from input
name = input("What's your name? ").strip().title()

# Say hello to user
print("hello,",name)
print("hello," + name) 
print(f"hello, {name}")
