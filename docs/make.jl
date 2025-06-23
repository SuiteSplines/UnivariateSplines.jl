using UnivariateSplines
using Documenter

DocMeta.setdocmeta!(UnivariateSplines, :DocTestSetup, :(using UnivariateSplines; using IgaBase); recursive=true)

makedocs(;
    modules=[UnivariateSplines],
    authors="René Hiemstra, Michał Mika and contributors",
    sitename="UnivariateSplines.jl",
    format=Documenter.HTML(;
        canonical="https://SuiteSplines.github.io/UnivariateSplines.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/SuiteSplines/UnivariateSplines.jl",
    devbranch="main",
)
