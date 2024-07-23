import tkinter as tk

window = tk.Tk()

# Add a widget
greeting = tk.Label(text="Hello, Tkinter")
greeting.pack()

label = tk.Label(
    text="Hello, Tkinter",
    fg="white",
    bg="black",
    width=10,
    height=10
)
label.pack()

# Always run for showing
window.mainloop()