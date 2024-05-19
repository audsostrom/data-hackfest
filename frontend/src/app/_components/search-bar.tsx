"use client"
import { useState, useEffect } from 'react';
import TextField from '@mui/material/TextField';
import Stack from '@mui/material/Stack';
import Autocomplete from '@mui/material/Autocomplete';
import { api } from '~/trpc/react';
import { RedirectType, redirect } from 'next/navigation'
import { revalidatePath } from 'next/cache'
import { Movie } from '~/server/db/schema';
import SearchIcon from '@mui/icons-material/Search';
import { usePathname, useSearchParams } from 'next/navigation'

function useDebounce(value, delay) {
  const [debouncedValue, setDebouncedValue] = useState(value)

  const pathname = usePathname()
  const searchParams = useSearchParams()
 
  useEffect(() => {
    const url = `${pathname}?${searchParams}`
    console.log(url)
    // You can now use the current URL
    // ...
  }, [pathname, searchParams])

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value)
    }, delay)

    return () => {
      clearTimeout(handler)
    }
  }, [value, delay])

  return debouncedValue
}


export default function SearchBar() {
  const [selectedMovie, setSelectedMovie] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const debouncedSearch = useDebounce(searchQuery, 500)
  const { data: movies, isLoading: isSearching } = api.movie.searchFor.useQuery(debouncedSearch);
  const movieOpts = movies ?? []
  useEffect(() => {
    if (debouncedSearch) {
      fetch(`/api/search?q=${debouncedSearch}`)
    }
  }, [debouncedSearch])
  return (
    <Stack spacing={2} sx={{ width: 300 }}>
      <Autocomplete
        id="movies"
        freeSolo
        options={movieOpts.map((option) => option.title)}
        value={selectedMovie}
        renderInput={(params) => 
        <TextField {...params} 
        // onClick={
        //   ()=> {
        //     console.log("tests", params);
        //     redirect('/movie/' + params.id);
        //   }
        // }
          // redirct.push({
          //   pathname: '/movie', 
          //   query: { movieId: params.id }
          // })
        value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} label="" />
      }
      onChange={(event, title) => {
        console.log(title);
        const selTitle = title ?? ""
        const selMovie = movieOpts.find((movie)=> movie.title == selTitle)
        // router.push({
        //     pathname: '/movie', 
        //     query: { movieId: selMovie?.id }
        //   })
        // revalidatePath('/movie');
        console.log(`/movie/${selMovie?.id}`)
        redirect(`/movie/${selMovie?.id}`, RedirectType.push);
      }}
      />
      {/* <Autocomplete
        freeSolo
        id="movies-2"
        disableClearable
        options={movieOpts.map((option) => option.title)}
        renderInput={(params) => (
          <TextField
            {...params}
            label="Search input"
            InputProps={{
              ...params.InputProps,
              type: 'search',
            }}
          />
        )}
      /> */}
    </Stack>
  );
}
