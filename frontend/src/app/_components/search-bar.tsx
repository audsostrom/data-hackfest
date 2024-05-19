"use client"
import { useState } from 'react';
import TextField from '@mui/material/TextField';
import Stack from '@mui/material/Stack';
import Autocomplete from '@mui/material/Autocomplete';
import { api } from '~/trpc/react';

export default function SearchBar() {
  const [searchQuery, setSearchQuery] = useState("");
  const { data: movies, isLoading: isSearching } = api.movie.searchFor.useQuery(searchQuery);
  const movieOpts = movies ?? []
  return (
    <Stack spacing={2} sx={{ width: 300 }}>
      <Autocomplete
        id="movies"
        freeSolo
        options={movieOpts.map((option) => option.title)}
        renderInput={(params) => 
        <TextField {...params} value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} label="movies" />
      }
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
