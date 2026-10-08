# OrganelleVerse R kernel worker. Implements the same length-prefixed
# protocol as the Python worker: per request one code frame, one inputs
# manifest frame, one outputs manifest frame (manifest entries are
# length-prefixed "name=path" lines); responds with one length-prefixed
# JSON object. Base R only - no packages, no networking.

protocol.input <- file("stdin", "rb")
# R's stdout() is a text terminal; the POSIX device exposes its binary stream.
protocol.output <- file("/dev/stdout", "wb", raw = TRUE)

read.frame <- function() {
  header <- readBin(protocol.input, "integer", n = 1, size = 4, endian = "big")
  if (length(header) == 0 || is.na(header)) return(NULL)
  raw <- readBin(protocol.input, "raw", n = header)
  rawToChar(raw)
}

read.manifest <- function() {
  count <- readBin(protocol.input, "integer", n = 1, size = 4, endian = "big")
  entries <- list()
  if (is.na(count)) return(entries)
  for (i in seq_len(count)) {
    line <- read.frame()
    if (is.null(line)) break
    parts <- strsplit(line, "=", fixed = TRUE)[[1]]
    entries[[parts[1]]] <- paste(parts[-1], collapse = "=")
  }
  entries
}

json.escape <- function(s) {
  s <- gsub("\\", "\\\\", s, fixed = TRUE)
  s <- gsub('"', '\\"', s, fixed = TRUE)
  s <- gsub("\r", "\\r", s, fixed = TRUE)
  s <- gsub("\n", "\\n", s, fixed = TRUE)
  s <- gsub("\t", "\\t", s, fixed = TRUE)
  paste0('"', s, '"')
}

write.frame <- function(text) {
  bytes <- charToRaw(enc2utf8(text))
  writeBin(as.integer(length(bytes)), protocol.output, size = 4, endian = "big")
  writeBin(bytes, protocol.output)
  flush(protocol.output)
}

repeat {
  code <- read.frame()
  if (is.null(code)) break
  inputs <- read.manifest()
  outputs <- read.manifest()
  input.paths <<- unlist(inputs)
  output.paths <<- unlist(outputs)
  captured <- character(0)
  error <- NULL
  zz <- textConnection("captured", "wr", local = TRUE)
  ok <- TRUE
  tryCatch(
    {
      sink(zz, split = FALSE)
      for (expression in parse(text = code)) {
        evaluated <- withVisible(eval(expression, envir = .GlobalEnv))
        if (evaluated$visible) print(evaluated$value)
      }
    },
    error = function(e) {
      ok <<- FALSE
      error <<- paste0(class(e)[1], ": ", conditionMessage(e))
    },
    finally = {
      sink(NULL)
      close(zz)
    }
  )
  stdout.text <- paste(captured, collapse = "\n")
  payload <- paste0(
    '{"ok":', if (ok) "true" else "false",
    ',"stdout":', json.escape(stdout.text),
    ',"result":null,"error":', if (ok) "null" else json.escape(error),
    "}"
  )
  write.frame(payload)
}
