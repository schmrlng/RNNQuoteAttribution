using StatsBase

word_counts = countmap(ASCIIString[])
open("futurama/futurama.txt") do f
    for line in eachline(f)
        addcounts!(word_counts, map(ascii, split(strip(split(line, '\t')[3]), ' ')))
    end
end
open("prideprejudice/prideprejudice.txt") do f
    for line in eachline(f)
        addcounts!(word_counts, map(ascii, split(strip(split(line, '\t')[3]), ' ')))
    end
end
addcounts!(word_counts, ["unk"])

@show length(word_counts)

for glove in ["glove.6B.100d.txt", "glove.6B.200d.txt", "glove.6B.300d.txt", "glove.6B.50d.txt", "glove.840B.300d.txt"]
    open(replace(glove, "txt", "filtered.txt"), "w") do writefiltered
        # ghetto two passes
        ct = 0
        open(glove) do f
            for line in eachline(f)
                word = split(line, ' ')[1]
                haskey(word_counts, word) && (ct += 1)
            end
        end
        
        @show "$ct $(split(glove, '.')[3][1:end-1])\n"
        write(writefiltered, "$ct $(split(glove, '.')[3][1:end-1])\n")
        
        open(glove) do f
            for line in eachline(f)
                word = split(line, ' ')[1]
                haskey(word_counts, word) && write(writefiltered, line)
            end
        end
    end
end