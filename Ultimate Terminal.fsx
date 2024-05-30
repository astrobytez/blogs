(**
# Install Alacritty with Fish on Fedora

In this post I will go through the steps to install Alacritty with the Fish shell on Fedora.
Additionally I setup the shell to display the system info using Neofetch and also a nice system prompt using Starship.

To begin open a bash terminal, for example Konsole. Usuall the default is to just opress Ctrl-Alt-T
Then run the following commands to install the required packages. Note you'll need to run the dnf commands as root (prefix with sudo).

``` bash
$ dnf copr enable atim/starship
$ dnf install -y alacritty fish starship neofetch

$ mkdir -p ~/.config/alacritty
$ nano ~/.config/alacritty/alacritty.toml
```

Then in the alacritty.toml file enter the following contents (refer to https://alacritty.org/config-alacritty.html for details):

```

[shell]
program = "/usr/bin/fish"

[window]
opacity = 0.8 # Optional
```

Once the changes are made press Ctrl-X then Y to save and quit back to bash.

You can go to system settings and set Alacritty as the default terminal and configure keyboard shortcuts for Ctrl-Shift-T

Next we can configure fish to run neofetch on startup:

``` bash
$ echo neofetch >> ~/.config/fish/config.fish
```

To use starship we need a NerdFont installed on the system, I recommend the JetBrains Mono fonts:
https://github.com/ryanoasis/nerd-fonts/releases/download/v3.2.1/JetBrainsMono.zip

The full selection of fonts can be found at https://www.nerdfonts.com/font-downloads

Once you have a suitable nerdfont you can unzip the bundle and install the fonts you desire.
Then we configure the starship prompt:
``` bash
$ echo "starship init fish | source" >> ~/.config/fish/config.fish
```

Now for the finishing touches, I really like the prompt design from Garuda Linux.
You can copy the same theme from the following https://github.com/inalireza/garuda-starship.toml

Copy the contents of the file to ~/.config/starship.toml, restart alacritty and you're good to go!

Enjoy!

*)
