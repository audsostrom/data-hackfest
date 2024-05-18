import React from 'react';
import { Box, Typography, Avatar, Divider, Grid } from '@mui/material';

export default function ProfileCard() {
    return (
        <Box
            sx={{
                borderRadius: 2,
                overflow: 'hidden',
                boxShadow: 3,
                bgcolor: 'background.paper',
                width: {
                    xs: '100%',
                    md: '90%',
                }
            }}
        >
            <Box
                sx={{
                    height: 100,
                    backgroundImage: 'url(https://m.media-amazon.com/images/M/MV5BYzI2YmZhZTQtMWQ1OC00YTMxLWJlNzgtYjYyZGZjOWEyNmQxXkEyXkFqcGdeQWFybm8@._V1_.jpg)',
                    backgroundSize: 'cover',
                    backgroundPosition: 'center',
                }}
            />
            <Box
                sx={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    p: 2,
                    marginTop: '-50px',
                }}
            >
                <Avatar sx={{ width: 92, height: 92, mb: 1, border: '6px solid white' }} />
                <Typography variant="h6">Audrey Something</Typography>
                <Typography variant="body2" color="text.secondary">
                    @audreysomething
                </Typography>
            </Box>
            <Grid container columns={13} justifyContent={'space-between'} sx={{
                paddingX: '10px',
                paddingBottom: 6,
            }}>
                <Grid item xs={4}>
                    <Box sx={{ textAlign: 'center', p: 1 }}>
                        <Typography variant="h6">18</Typography>
                        <Typography variant="body2">Movies Seen</Typography>
                    </Box>
                </Grid>
                <Divider orientation="vertical" variant="middle" flexItem />
                <Grid item xs={4}>
                    <Box sx={{ textAlign: 'center', p: 1 }}>
                        <Typography variant="h6">18</Typography>
                        <Typography variant="body2">Friends</Typography>
                    </Box>
                </Grid>
                <Divider orientation="vertical" variant="middle" flexItem />
                <Grid item xs={4}>
                    <Box sx={{ textAlign: 'center', p: 1 }}>
                        <Typography variant="h6">18</Typography>
                        <Typography variant="body2">Bucket Lists</Typography>
                    </Box>
                </Grid>
            </Grid>
        </Box>
    );
}
