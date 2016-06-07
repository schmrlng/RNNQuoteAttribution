using StatsBase

function remove_modifier(name)
    name = ascii(name)
    modstart = max(findfirst(name, '('), findfirst(name, '['))
    if modstart > 0
        name = name[1:modstart-1]
    end
    lowercase(strip(name))
end

speaker_counts = countmap(ASCIIString[])
word_counts = countmap(ASCIIString[])
open("futurama.txt", "w") do writefile
    for (fi, fname) in enumerate(readdir("rawdata"))
        @show fname
        open("rawdata/" * fname) do f
            s = readstring(f)
            lines = eachmatch(r"</b><b>                                     (.+)\n</b>((.*\n)+?)<b>", s)
            speakers = collect(remove_modifier(l[1]) for l in lines)
            addcounts!(speaker_counts, speakers)
            quotes = collect(replace(l[2], r"<b>|</b>", "") for l in lines)
            allquotes = join(quotes, " delimdelimdelim ")
            tokenized = map(strip, split(
                readstring(pipeline(`echo $allquotes`, `java edu.stanford.nlp.process.PTBTokenizer -lowerCase -options ptb3Escaping=false`))
                , "delimdelimdelim"))
            for q in tokenized
                if q == ""
                    warn("empty quote in $fname")
                else
                    addcounts!(word_counts, map(ascii, split(q)))
                end
            end
            @assert length(speakers) == length(tokenized)
            for (s,q) in zip(speakers, tokenized)
                write(writefile, string(fi) * "\t" * s * "\t" * strip(replace(q, "\n", " ")) * "\n")
            end
            # quotes = map(q -> " " * strip(replace(q, "\n", " ")) * " ", tokenized)
        end
    end
end

(sort(collect(speaker_counts), by=last), sort(collect(word_counts), by=last))